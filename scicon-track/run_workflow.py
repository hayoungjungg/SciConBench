"""Prefect workflow for the monthly SciConBench-Track pipeline.

Run once:      python scicon-track/run_workflow.py --once
Run on a 30-day schedule: python scicon-track/run_workflow.py
Limit to N DOIs (smoke test): python scicon-track/run_workflow.py --once --max-dois 5
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make scicon-track/ sub-packages (config, db, huggingface) importable,
# and scripts/ sub-packages (data_preprocessing, data_labeling, data_collection).
_track_dir = str(Path(__file__).resolve().parent)
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
for _p in (_track_dir, _scripts_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import asyncio
import os
import smtplib
import traceback
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from itertools import islice
from typing import List, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from prefect import flow, task
from prefect.tasks import exponential_backoff

load_dotenv()


# ── Notification helpers ───────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(ZoneInfo("America/New_York")).strftime("%a, %d %b %Y %H:%M:%S EST")


def _send_email(subject: str, body: str) -> None:
    try:
        sender = os.environ["EMAIL_SENDER"]
        password = os.environ["EMAIL_APP_PASSWORD"]
        recipient = os.environ["EMAIL_RECIPIENT"]
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.ehlo()
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        print(f"Email sent: {subject}")
    except Exception as exc:
        print(f"Email notification failed (pipeline continues): {exc}")


@task(name="Notify: pipeline started")
def task_notify_start(started_at: str) -> None:
    _send_email(
        subject=f"[SciConBench-Track] Started — {started_at}",
        body=f"Monthly pipeline started at {started_at}.",
    )


@task(name="Notify: pipeline succeeded")
def task_notify_success(started_at: str, ended_at: str) -> None:
    _send_email(
        subject=f"[SciConBench-Track] Completed — {ended_at}",
        body=f"Pipeline completed successfully.\n\nStarted:  {started_at}\nFinished: {ended_at}",
    )


@task(name="Notify: pipeline failed")
def task_notify_error(started_at: str, error: str) -> None:
    _send_email(
        subject=f"[SciConBench-Track] FAILED — {started_at}",
        body=f"Pipeline failed.\n\nStarted: {started_at}\n\n--- Error ---\n{error}",
    )


# ── Pipeline tasks ─────────────────────────────────────────────────────────────


@task(name="Initialize database")
def task_init_db() -> None:
    from db import init_db
    init_db(force=False)


@task(name="Register core set from HuggingFace")
def task_register_core_set() -> list[str]:
    from data_collection.collector import DataCollector
    return DataCollector().register_core_set()


@task(name="Discover new rolling reviews")
def task_discover_rolling(cohort_month: str, max_dois: int | None = None) -> list[str]:
    """Return DOIs for reviews not yet in the HuggingFace benchmark.

    Uses the HF benchmark for what is already known.  
    New DOIs are registered in the DB as ROLLING for downstream stages.
    """
    from config import hf_cfg
    from datasets import load_dataset
    from data_collection.collector import DataCollector
    from db.db import DOIInfo, PanelType, ProcessingStatus
    from db import Session
    from datetime import datetime as _dt

    # Load HF benchmark as the authoritative known-DOI set
    src = hf_cfg.source
    ds = load_dataset(src.repo_id, src.config, split=src.split)
    hf_dois = {row["doi"] for row in ds}
    print(f"  HuggingFace benchmark: {len(hf_dois):,} known DOIs")

    new_dois = DataCollector().discover_rolling_reviews(known_dois=hf_dois)
    if max_dois is not None:
        new_dois = new_dois[:max_dois]

    # Register new DOIs in DB
    if new_dois:
        with Session() as session:
            existing = {row.doi for row in session.query(DOIInfo.doi).all()}
            added = 0
            for doi in new_dois:
                if doi not in existing:
                    session.add(DOIInfo(
                        doi=doi,
                        panel_type=PanelType.ROLLING,
                        processing_status=ProcessingStatus.REGISTERED,
                        added_at=_dt.utcnow(),
                    ))
                    added += 1
            session.commit()
        print(f"  Registered {added} new rolling DOIs in DB")

    return new_dois


@task(name="Download PDFs via Wiley TDM")
def task_download_pdfs(dois: list[str]) -> dict[str, str]:
    from data_collection.collector import DataCollector
    return DataCollector().download_pdfs(dois)


@task(name="Extract reference text from PDFs")
def task_extract_text(dois: list[str] | None = None) -> int:
    from data_collection.collector import DataCollector
    return DataCollector().extract_and_store_text(dois)


@task(
    name="Generate questions",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_questions(dois: list[str] | None = None) -> None:
    from data_preprocessing.question_generation import QuestionGenerator
    from db.db import ProcessingStatus
    from db.utils import (
        get_dois_needing_question,
        get_reviews_from_db,
        populate_questions,
        update_doi_status,
    )

    target_dois = set(dois) if dois else get_dois_needing_question()
    if not target_dois:
        print("No DOIs need question generation.")
        return

    generator = QuestionGenerator()
    import re as _re

    def _parse_background(text: str) -> str:
        """Extract Background or Rationale section for use as background_context."""
        for heading in (r"Background", r"Rationale"):
            m = _re.search(
                rf"(?m)^{heading}\s*\n(.*?)(?=\n[A-Z][a-zA-Z'\u2019 ]+\n|\n\[PLAIN|\Z)",
                text, _re.DOTALL,
            )
            if m:
                return m.group(1).strip()
        return ""

    reviews = get_reviews_from_db()
    for doi in target_dois:
        review = reviews.get(doi)
        if not review:
            continue
        # Use the structured objectives section when available; fall back to full reference text.
        objective_text = review.get("objectives") or review.get("reference_text") or ""
        if not objective_text:
            continue
        ref_text = review.get("reference_text") or ""
        background_ctx = _parse_background(ref_text)
        try:
            result = generator.run(
                objective=objective_text,
                background_context=background_ctx,
            )
            question = result.get("question", "")
            if question:
                populate_questions(doi, question)
                update_doi_status(doi, ProcessingStatus.QUESTION_GENERATED)
        except Exception as exc:
            print(f"Question generation failed for {doi}: {exc}")


@task(
    name="Generate Cochrane atomic facts (batch)",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_cochrane_facts_batch(batch: dict, questions: dict) -> None:
    from data_preprocessing.atomic_fact_generation import AtomicFactGenerator
    from db.db import ProcessingStatus
    from db.utils import populate_atomic_facts, update_doi_status

    generator = AtomicFactGenerator()
    for doi, review in batch.items():
        question = questions.get(doi, "")
        # Use authors' conclusions section when available; fall back to full reference text.
        generation_text = review.get("authors_conclusions") or review.get("reference_text", "")
        if not generation_text or not question:
            continue
        try:
            facts_pairs, _para_breaks, _meta = generator.run(
                generation=generation_text,
                question=question,
            )
            populate_atomic_facts(doi=doi, source="cochrane", atomic_facts_pairs=facts_pairs)
            update_doi_status(doi, ProcessingStatus.FACTS_GENERATED)
        except Exception as exc:
            print(f"Cochrane atomic facts failed for {doi}: {exc}")


@task(
    name="Run model queries",
    retries=2,
    retry_delay_seconds=exponential_backoff(backoff_factor=15),
)
async def task_run_queries(dois: list[str], run_month: str) -> None:
    """Query all configured model configs for the given DOIs."""
    import re
    from config import query_cfg
    from db.utils import get_questions, get_reviews_from_db, populate_model_response
    from sciconharness import SciConHarness
    from sciconharness.utils.hf_benchmark_cache import (
        load_cochrane_titles_cached,
        load_doi_to_title_cached,
        standardize_title,
    )

    # Load clean-room filtering variables from the local HF benchmark cache
    # (see sciconharness/utils/hf_benchmark_cache.py) instead of pulling the
    # full dataset again here — task_upload_to_hf (the previous pipeline
    # stage) already refreshed this cache from the exact rows it just
    # merged/published, so this is just a disk read, no second HF pull.
    all_titles = load_cochrane_titles_cached()
    doi_to_title = load_doi_to_title_cached()

    reviews = get_reviews_from_db()
    questions_map = get_questions()

    # Also include rolling DOIs not yet in the HF dataset (i.e. not yet
    # covered by task_upload_to_hf's just-refreshed cache — e.g. brand new
    # this month and still missing a question/atomic-facts, which
    # get_sciconbench_rows() requires before a DOI is eligible for upload).
    from db.utils import get_all_dois
    for doi in dois:
        if doi not in doi_to_title:
            review = reviews.get(doi, {})
            if review.get("name"):
                doi_to_title[doi] = review["name"]
                all_titles.append(standardize_title(review["name"]))

    HARNESS_CONFIGS = [
        ("no_tools",     False, False),
        ("tools",        True,  False),
        ("tools_filter", True,  True),
    ]

    def _extract_conclusion(text: str) -> str:
        m = re.search(r"\[\[\[(.*?)\]\]\]", text or "", re.DOTALL)
        return m.group(1).strip() if m else (text or "").strip()

    # NOTE: api_key / base_url / api_version are intentionally left unset here
    # (None) for every provider rather than force-passing OpenAI/Azure OpenAI
    # credentials (AZURE_OPENAI_KEY / OPENAI_BASE_URL). SciConHarness's
    # create_provider() (sciconharness/utils/query_utils.py) already resolves
    # the right credentials per-provider from env vars — e.g. DeepSeek-V4-Pro
    # (provider="azure") uses its own dedicated COCHRANE_DASHBOARD_OPENAI_KEY /
    # COCHRANE_DASHBOARD_BASE_URL, OpenRouter models (Kimi/GLM/Qwen) use
    # OPENROUTER_API_KEY (or the filtering/base-model variants selected via
    # enable_filtering/enable_tool_calling), Gemini falls back to Vertex AI,
    # and Claude auto-detects Foundry — all of which would be silently
    # bypassed if a non-None api_key/base_url were forced through for every
    # provider in this loop (as previously done here for only OpenAI).
    for provider, model in query_cfg.iter_models():
        for config_label, use_tools, use_filter in HARNESS_CONFIGS:
            harness = SciConHarness(
                provider=provider,
                model=model,
                enable_tools=use_tools,
                enable_filtering=use_filter,
                cochrane_titles=all_titles,
                doi_to_title=doi_to_title,
                save_results=False,
            )
            async with harness:
                for doi in dois:
                    question = questions_map.get(doi)
                    review = reviews.get(doi, {})
                    pub_date_raw = review.get("publication_date")
                    pub_date_str = (
                        pub_date_raw.strftime("%d %B %Y")
                        if pub_date_raw and hasattr(pub_date_raw, "strftime")
                        else str(pub_date_raw or "")
                    )
                    if not question:
                        continue
                    try:
                        response, usage = await harness.query(
                            question,
                            doi=doi,
                            publication_date=pub_date_str,
                        )
                        populate_model_response(
                            doi=doi,
                            model=model,
                            provider=provider,
                            config_label=config_label,
                            query=question,
                            response=response,
                            token_usage=usage,
                            enable_tool_calling=use_tools,
                            enable_filtering=use_filter,
                            publication_date=pub_date_raw,
                            run_month=run_month,
                        )
                    except Exception as exc:
                        print(f"Query failed for doi={doi} provider={provider} config={config_label}: {exc}")


@task(
    name="Generate model-response atomic facts (batch)",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_generate_response_facts_batch(batch: dict) -> None:
    import re
    from data_preprocessing.atomic_fact_generation import AtomicFactGenerator
    from db.utils import populate_atomic_facts

    def _extract_conclusion(text: str) -> str:
        m = re.search(r"\[\[\[(.*?)\]\]\]", text or "", re.DOTALL)
        return m.group(1).strip() if m else (text or "").strip()

    generator = AtomicFactGenerator()
    for _response_id, resp_data in batch.items():
        doi = resp_data["doi"]
        question = resp_data.get("query", "")
        conclusion = _extract_conclusion(resp_data.get("response", ""))
        if not conclusion or not question:
            continue
        try:
            facts_pairs, _para_breaks, _meta = generator.run(
                generation=conclusion,
                question=question,
            )
            populate_atomic_facts(doi=doi, source="model_response", atomic_facts_pairs=facts_pairs)
        except Exception as exc:
            print(f"Response atomic facts failed for doi={doi}: {exc}")


@task(
    name="Run factual precision analysis",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_run_precision(run_month: str | None = None) -> None:
    import os
    from config import llm_judge_cfg
    from data_labeling import FactualPrecisionAnalyzer, ModelJudgeFactory
    from data_labeling.utils.prompts import (
        factual_precision_few_shot_prompt,
        factual_precision_zero_shot_prompt,
    )
    from db.utils import (
        get_all_model_responses,
        get_atomic_facts,
        get_reviews_from_db,
        populate_precision_result,
    )

    j = llm_judge_cfg["llm_judge"]
    prec = llm_judge_cfg["factual_precision_analyzer"]

    judge = ModelJudgeFactory.create(
        model=j["model"],
        api_key=os.environ.get(j["api_key_variable"]),
        base_url=os.environ.get(j["base_url_variable"]),
        openai_reasoning_effort=j.get("openai_reasoning_effort", "none"),
    )
    prompt = (
        factual_precision_few_shot_prompt
        if prec.get("prompt_mode", "few-shot") == "few-shot"
        else factual_precision_zero_shot_prompt
    )
    os.environ["DATA_LABELING_JUDGE_DELAY_S"] = str(prec.get("delay", 0))
    precision_judge = FactualPrecisionAnalyzer(
        llm_judge=judge,
        precision_prompt_template=prompt,
        temperature=prec.get("temperature", 0.2)
    )

    model_response_facts = get_atomic_facts("model_response")
    reviews = get_reviews_from_db()
    all_responses = get_all_model_responses(run_month=run_month)

    for response_id, resp_data in all_responses.items():
        doi = resp_data["doi"]
        model_facts = model_response_facts.get(doi, {}).get("all_facts", [])
        reference_text = reviews.get(doi, {}).get("reference_text", "")
        if not model_facts or not reference_text:
            continue
        try:
            result = precision_judge.compute_factual_precision(
                llm_atomic_facts=model_facts,
                ground_truth_text=reference_text,
            )
            populate_precision_result(response_id, doi, result)
        except Exception as exc:
            print(f"Precision failed for doi={doi}: {exc}")


@task(
    name="Run factual recall analysis",
    retries=3,
    retry_delay_seconds=exponential_backoff(backoff_factor=10),
)
def task_run_recall(run_month: str | None = None) -> None:
    import os
    import re
    from config import llm_judge_cfg
    from data_labeling import FactualRecallAnalyzer, ModelJudgeFactory
    from data_labeling.utils.prompts import (
        factual_recall_few_shot_prompt,
        factual_recall_zero_shot_prompt,
    )
    from db.utils import (
        get_all_model_responses,
        get_atomic_facts,
        populate_recall_result,
    )

    def _extract_conclusion(text: str) -> str:
        m = re.search(r"\[\[\[(.*?)\]\]\]", text or "", re.DOTALL)
        return m.group(1).strip() if m else (text or "").strip()

    j = llm_judge_cfg["llm_judge"]
    rec = llm_judge_cfg["factual_recall_analyzer"]

    judge = ModelJudgeFactory.create(
        model=j["model"],
        api_key=os.environ.get(j["api_key_variable"]),
        base_url=os.environ.get(j["base_url_variable"]),
        openai_reasoning_effort=j.get("openai_reasoning_effort", "none"),
    )
    prompt = (
        factual_recall_few_shot_prompt
        if rec.get("prompt_mode", "zero-shot") == "few-shot"
        else factual_recall_zero_shot_prompt
    )
    os.environ["DATA_LABELING_JUDGE_DELAY_S"] = str(rec.get("delay", 0))
    recall_judge = FactualRecallAnalyzer(
        llm_judge=judge,
        recall_prompt_template=prompt,
        temperature=rec.get("temperature", 1.0)
    )

    cochrane_facts = get_atomic_facts("cochrane")
    all_responses = get_all_model_responses(run_month=run_month)

    for response_id, resp_data in all_responses.items():
        doi = resp_data["doi"]
        conclusion = _extract_conclusion(resp_data.get("response", ""))
        gt_facts = cochrane_facts.get(doi, {}).get("all_facts", [])
        if not conclusion or not gt_facts:
            continue
        try:
            result = recall_judge.compute_factual_recall(
                llm_response_text=conclusion,
                article_atomic_facts=gt_facts,
            )
            populate_recall_result(response_id, doi, result)
        except Exception as exc:
            print(f"Recall failed for doi={doi}: {exc}")


@task(name="Upload to HuggingFace")
def task_upload_to_hf() -> None:
    from huggingface.uploader import SciConBenchUploader
    uploader = SciConBenchUploader()
    uploader.save_to_parquet()
    # Regenerate sciconharness's local title/DOI-mapping filter caches from
    # the same merged rows before publishing, so they never lag the dataset
    # we're about to push (see sciconharness/utils/hf_benchmark_cache.py).
    uploader.refresh_filter_caches()
    uploader.upload()


# ── Batch helpers ──────────────────────────────────────────────────────────────


def _batches(d: dict, size: int) -> List[Tuple[int, int]]:
    n = len(d)
    num = (n + size - 1) // size
    return [(i * size, min((i + 1) * size, n)) for i in range(num)]


# ── Main flow ──────────────────────────────────────────────────────────────────


@flow(name="SciConBench-Track Monthly Pipeline", log_prints=True)
def sciconbench_track_pipeline(
    batch_size: int = 500,
    max_dois: int | None = None,
) -> None:
    """End-to-end monthly pipeline.

    Stages:
      1.  Initialize DB
      2.  Register core set from HuggingFace
      3.  Discover new rolling reviews (diff against HuggingFace benchmark)
      4.  Download rolling PDFs via Wiley TDM
      5.  Extract reference text from PDFs
      6.  Generate clinical questions for rolling reviews
      7.  Generate Cochrane atomic facts for rolling reviews
      8.  Upload new reviews to HuggingFace (benchmark data only)
      9.  Query all models on all reviews
      10. Generate model-response atomic facts
      11. Run precision & recall analysis
    """
    from db.utils import (
        get_dois_by_panel,
        get_dois_by_status,
        get_questions,
        get_reviews_from_db,
        get_unprocessed_model_responses,
    )
    from db.db import PanelType, ProcessingStatus

    cohort_month = date.today().strftime("%Y-%m")
    started_at = _now()
    task_notify_start(started_at)

    try:
        # 1. Init DB
        task_init_db()

        # 2. Core set (idempotent)
        core_dois = task_register_core_set()

        # 3. Discover rolling reviews
        rolling_dois = task_discover_rolling(cohort_month=cohort_month, max_dois=max_dois)

        # 4. Download PDFs for new rolling reviews
        task_download_pdfs(rolling_dois)

        # 5. Extract text
        task_extract_text()

        # 6. Generate questions for rolling reviews without one
        task_generate_questions()

        # 7. Cochrane atomic facts for rolling reviews
        rolling_needing_facts = list(get_dois_by_status(ProcessingStatus.QUESTION_GENERATED))
        reviews = get_reviews_from_db()
        questions_map = get_questions()
        processed_facts = set(get_atomic_facts("cochrane").keys())
        to_process = {
            doi: reviews[doi] for doi in rolling_needing_facts
            if doi in reviews and doi not in processed_facts
        }
        for start, end in _batches(to_process, batch_size):
            batch = dict(islice(to_process.items(), start, end))
            task_generate_cochrane_facts_batch(batch, questions_map)

        # 8. Upload new reviews to HuggingFace (benchmark data: reviews + questions + facts)
        task_upload_to_hf()

        # 9. Query all models (core + rolling)
        all_dois = list(get_dois_by_panel(PanelType.CORE)) + rolling_dois
        if max_dois is not None:
            all_dois = all_dois[:max_dois]

        asyncio.run(task_run_queries(all_dois, run_month=cohort_month))

        # 10. Model-response atomic facts
        unprocessed = get_unprocessed_model_responses(run_month=cohort_month)
        for start, end in _batches(unprocessed, batch_size):
            batch = dict(islice(unprocessed.items(), start, end))
            task_generate_response_facts_batch(batch)

        # 11. Precision & recall
        task_run_precision(run_month=cohort_month)
        task_run_recall(run_month=cohort_month)

        task_notify_success(started_at, _now())

    except Exception:
        task_notify_error(started_at, traceback.format_exc())
        raise


# ── Entry point ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SciConBench-Track Monthly Pipeline")
    parser.add_argument("--once", action="store_true",
                        help="Run immediately once instead of starting the scheduler.")
    parser.add_argument("--max-dois", type=int, default=None, metavar="N",
                        help="Limit to N DOIs (for smoke-testing).")
    parser.add_argument("--batch-size", type=int, default=500, metavar="N",
                        help="Atomic-fact batch size.")
    args = parser.parse_args()

    if args.once:
        sciconbench_track_pipeline(batch_size=args.batch_size, max_dois=args.max_dois)
    else:
        sciconbench_track_pipeline.serve(
            name="sciconbench-track-monthly",
            interval=timedelta(days=30),
        )
