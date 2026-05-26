#!/usr/bin/env python3
"""
SciConBench — End-to-End Evaluation Example
============================================

Demonstrates the full pipeline from raw data to SciConBench performance metrics:

  Step 1 · Load the HuggingFace benchmark dataset
  Step 2 · Explore the dataset and create variables for clean-room filtering
  Step 3 · Query a model via SciConHarness (clean-room protocol)
  Step 4 · Generate atomic facts from model conclusions
  Step 5 · Label factual precision and recall
  Step 6 · Compute aggregate metrics from labeled facts

Prerequisites
-------------
  pip install -e .

  API keys in .env (copy .env.example → .env and fill in):
    OPENAI_API_KEY / AZURE_OPENAI_KEY + OPENAI_BASE_URL   (for OpenAI models + atomic fact generation + LLM judge labeling)
    ANTHROPIC_API_KEY / AZURE_ANTHROPIC_API_KEY + AZURE_ANTHROPIC_BASE_URL + AZURE_ANTHROPIC_RESOURCE_NAME  (for Claude)
    GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION          (for Gemini)
    SERPER_API_KEY + JINA_API_KEY + S2_API_KEY             (for web search and browsing tools in SciConHarness)

Usage
-----
  python evaluate.py                  # full pipeline (all steps)
  python evaluate.py --steps 1,2     # only load & explore dataset
  python evaluate.py --steps 3       # only run model queries (requires Step 1 to have run)
  python evaluate.py --n 5           # use 5 benchmark examples instead of 2
  python evaluate.py --model gemini-3-pro-preview --provider gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import textwrap
from pathlib import Path

import dotenv

dotenv.load_dotenv(Path(__file__).resolve().parent / ".env")

# Make scripts/ sub-packages (data_preprocessing, data_labeling) importable
# without requiring `pip install -e .` — works regardless of clone location.
_scripts_dir = str(Path(__file__).resolve().parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Number of benchmark examples to run through the pipeline.
N_SAMPLES = 2

# Model to evaluate.  Options per provider:
#   openai     →  gpt-5.1, o4-mini-deep-research, o3-deep-research, ...
#   claude     →  claude-sonnet-4-5
#   gemini     →  gemini-3-pro-preview...
#   perplexity →  sonar-reasoning-pro, sonar-pro, ...
MODEL    = "gpt-5.1"
PROVIDER = "openai"

# Directory for pre-computed Google Drive data (used in Step 6)
DATA_DIR = Path(__file__).parent / "data"

# HuggingFace dataset identifier
HF_DATASET = "hayoungjung/SciConBench"
HF_CONFIG  = "benchmark"
HF_SPLIT   = "test"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "═" * 72
DIV  = "─" * 72

def banner(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def section(title: str) -> None:
    print(f"\n{DIV}\n  {title}\n{DIV}")

def extract_conclusion(text: str) -> str:
    """Return the [[[conclusion]]] from a harness response, or the raw text."""
    m = re.search(r"\[\[\[(.*?)\]\]\]", text or "", re.DOTALL)
    return m.group(1).strip() if m else (text or "").strip()


# =============================================================================
# Step 1 — Load the HuggingFace benchmark dataset
# =============================================================================

def step1_load_dataset():
    banner("Step 1 · Load HuggingFace Benchmark Dataset")

    from datasets import load_dataset

    print(f"  Loading {HF_DATASET!r} (config={HF_CONFIG!r}, split={HF_SPLIT!r}) …")
    ds = load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)
    print(f"  {len(ds)} examples  |  columns: {ds.column_names}")

    return ds


# =============================================================================
# Step 2 — Explore the dataset and build clean-room filtering variables
# =============================================================================

def step2_explore(ds):
    banner("Step 2 · Explore Dataset & Build Clean-Room Filtering Variables")

    # ── 2a. Dataset overview ─────────────────────────────────────────────────
    section("2a. Single example — all fields")
    row = ds[0]
    for key, val in row.items():
        snippet = str(val)
        if len(snippet) > 110:
            snippet = snippet[:107] + "…"
        print(f"  {key:<22}: {snippet}")

    # ── 2b. Summary statistics ────────────────────────────────────────────────
    section("2b. Summary statistics")
    from collections import Counter
    years = [str(d).split()[-1] for d in ds["publication_date"]]
    by_year = dict(sorted(Counter(years).items()))
    print(f"  Total examples      : {len(ds)}")
    print(f"  Unique DOIs         : {len(set(ds['doi']))}")
    print(f"  Publication years   : {by_year}")

    # ── 2c. Filtering variables ────────────────────────────────────────────────
    #
    # Three variables drive the CochraneResultFilter clean-room protocol:
    #
    #   cochrane_titles — all review titles in the benchmark.
    #     The filter blocks any search/browse result whose page title or text
    #     matches one of these titles (prevents the agent from reading
    #     the Cochrane review being evaluated, or related reviews).
    #
    #   doi_to_title — DOI → title mapping.
    #     Tells the filter which *specific* title to protect for each query,
    #     so it can apply stricter blocking for the target review.
    #
    #   publication_date — the review's own publication date (per query).
    #     Results published after this date are suppressed to preserve the
    #     temporal integrity of the benchmark (no post-hoc evidence).
    #
    section("2c. Clean-room filtering variables (used in Steps 4 & 5)")
    all_titles   = list(ds["title"])
    doi_to_title = {r["doi"]: r["title"] for r in ds}
    doi_to_date  = {r["doi"]: r["publication_date"] for r in ds}

    print(f"  cochrane_titles ({len(all_titles)} entries)")
    for t in all_titles[:3]:
        print(f"    · {t[:80]}")
    print(f"    …")
    print(f"\n  doi_to_title   ({len(doi_to_title)} entries)")
    for doi, title in list(doi_to_title.items())[:3]:
        print(f"    {doi} → {title[:60]}")
    print(f"\n  doi_to_date    ({len(doi_to_date)} entries)")
    for doi, date in list(doi_to_date.items())[:3]:
        print(f"    {doi} → {date}")

    return all_titles, doi_to_title, doi_to_date


# =============================================================================
# Step 3 — Query a model via SciConHarness (clean-room protocol)
# =============================================================================

async def step3_query(ds, all_titles, doi_to_title, doi_to_date):
    """
    Runs N_SAMPLES benchmark questions through three harness configurations
    to illustrate the effect of enabling tools and the clean-room filter:

      no_tools      — plain LLM call, no web access, no filtering
      tools         — web search + browse enabled, no filtering
      tools_filter  — web search + browse + CochraneResultFilter (recommended)

    The CochraneResultFilter suppresses:
      · Cochrane review pages (title-matched or DOI-matched)
      · Any source published after the review's cut-off date
      · Cochrane.org derivatives (press releases, plain-language summaries)
    """
    banner("Step 3 · Query Model via SciConHarness")

    from sciconharness import SciConHarness

    samples = [ds[i] for i in range(N_SAMPLES)]

    section(f"Selected {N_SAMPLES} benchmark examples")
    for i, row in enumerate(samples, 1):
        print(f"\n  [{i}] {row['doi']}")
        print(f"      Title : {row['title'][:80]}")
        print(f"      Date  : {row['publication_date']}")
        print(f"      Q     : {textwrap.shorten(row['question'], width=100)}")

    CONFIGS = [
        ("no_tools",     False, False),
        ("tools",        True,  False),
        ("tools_filter", True,  True ),
    ]

    all_results: dict[str, list[dict]] = {}

    for config_label, use_tools, use_filter in CONFIGS:
        section(f"Config: {config_label}  (tools={use_tools}, filter={use_filter})")

        harness = SciConHarness(
            provider         = PROVIDER,
            model            = MODEL,
            # Azure OpenAI — pass api_key + base_url for Azure endpoints.
            # For standard OpenAI, omit base_url (reads OPENAI_API_KEY from env).
            api_key          = os.environ.get("AZURE_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url         = os.environ.get("OPENAI_BASE_URL"),
            api_version      = os.environ.get("OPENAI_API_VERSION", "2025-04-01-preview"),
            enable_tools     = use_tools,
            enable_filtering = use_filter,
            # Clean-room filter variables (required when enable_filtering=True):
            cochrane_titles  = all_titles,    # block all Cochrane review titles
            doi_to_title     = doi_to_title,  # identify which review is being queried
            save_results     = True,          # saves result.json to sciconharness/logs/
        )

        config_entries: list[dict] = []

        async with harness:
            for row in samples:
                doi      = row["doi"]
                question = row["question"]
                pub_date = row["publication_date"]

                print(f"\n  → {doi}")
                print(f"    Q: {textwrap.shorten(question, width=100)}")
                print(f"    Running …")

                response, usage = await harness.query(
                    question,
                    doi              = doi,
                    publication_date = pub_date,   # filter: block results after this date
                )

                conclusion = extract_conclusion(response)
                print(f"    Conclusion ({len(conclusion)} chars): "
                      f"{textwrap.shorten(conclusion, width=90)}")
                print(f"    Token usage: {usage}")

                config_entries.append({
                    "doi":        doi,
                    "question":   question,
                    "pub_date":   pub_date,
                    "response":   response,
                    "conclusion": conclusion,
                    "usage":      usage,
                })

        all_results[config_label] = config_entries

    return all_results


# =============================================================================
# Step 4 — Generate atomic facts from model conclusions
# =============================================================================

def step4_atomic_facts(all_results):
    """
    Decomposes each generated conclusion into independent, verifiable atomic facts
    using our LLM pipeline (AtomicFactGenerator).

    Pipeline stages:
      0. Sentence splitting          (NLTK / spaCy)
      1. Decomposition               (LLM: split each sentence into facts)
      2. Decontextualization         (LLM: resolve vague references)
      3. Incomplete fact detection   (LLM: rewrite dependent claims)
      4. Irrelevant fact filtering   (LLM: drop off-topic facts)
      5. Redundant fact filtering    (LLM: keep most atomic, non-duplicate set)

    Atomic facts are the unit of measurement for factual precision and recall.
    """
    banner("Step 4 · Generate Atomic Facts from Model Conclusions")

    from data_preprocessing.atomic_fact_generation import AtomicFactGenerator

    # Use default model configs (reads API keys from .env)
    generator = AtomicFactGenerator()

    config_to_doi_to_facts: dict[str, dict[str, list[str]]] = {}

    for config_label, entries in all_results.items():
        print(f"\n  Config: {config_label}")
        doi_to_facts: dict[str, list[str]] = {}

        for entry in entries:
            doi        = entry["doi"]
            conclusion = entry["conclusion"]
            question   = entry["question"]

            if not conclusion:
                print(f"  [{doi}] No conclusion available — skipping.")
                continue

            print(f"\n  Decomposing conclusion for: {doi}")
            print(f"  Input length: {len(conclusion)} chars")

            facts_pairs, _para_breaks, metadata = generator.run(
                generation                  = conclusion,
                question                    = question,
                verbose                     = False,   # set to True to print verbose comparisons
                enable_incomplete_detection = True,
                enable_irrelevant_filtering = True,
                enable_redundant_filtering  = True,
            )

            facts = [f for _sent, fs in facts_pairs for f in fs]
            doi_to_facts[doi] = facts

            tok = metadata.get("token_usage", {}) if isinstance(metadata, dict) else {}
            print(f"  → {len(facts)} atomic facts  (tokens: {tok})")

        config_to_doi_to_facts[config_label] = doi_to_facts

    return config_to_doi_to_facts


# =============================================================================
# Step 5 — Label factual precision and recall
# =============================================================================

def step5_label(ds, all_results, config_to_doi_to_facts):
    """
    Uses an LLM judge (gpt-5.4-mini) to label each atomic fact:

      Factual Precision — judges each generated fact (from model's conclusion) against 
      the ground-truth Cochrane review article (exposed as "reference_text" column in the SciConBench
      dataset on HuggingFace):
        SUPPORTED     model fact is supported by the ground truth
        CONTRADICTED   model fact conflicts with the ground truth
        NOT SUPPORTED      model fact cannot be verified from the ground truth

        Score = (E/T) × (1 – C/T)   [entailment rate × non-contradiction rate]

      Factual Recall — judges each *ground-truth* fact from the Authors' Conclusion in the
      Cochrane review article (exposed as "all_facts" column in the SciConBench dataset on 
      HuggingFace) against the generated conclusions:
        SUPPORTED     the model response covers this ground-truth fact
        NOT SUPPORTED the model response misses this ground-truth fact

        Score = supported / total_gt_facts

    The ground-truth reference text and facts come from the Cochrane review article;
    the HuggingFace dataset exposes them in the "reference_text" and
    "all_facts" columns.
    """
    banner("Step 5 · Label Factual Precision & Recall")

    from data_labeling import make_precision_judge, make_recall_judge
    from data_labeling.utils import compute_f1

    # Factory functions return FactualPrecisionAnalyzer / FactualRecallAnalyzer
    # pre-configured with the judge model and prompt template validated by clincal
    # experts and used in the paper.
    precision_judge = make_precision_judge()   # gpt-5.4-mini, few-shot, temp 0.2
    recall_judge    = make_recall_judge()      # gpt-5.4-mini, zero-shot, temp 1.0

    # Ground-truth lookups from the dataset
    # (column names depend on the HuggingFace dataset version)
    gt_text_col   = "reference_text"  if "reference_text"  in ds.column_names else None
    gt_facts_col  = "all_facts"       if "all_facts"       in ds.column_names else None

    if not gt_text_col:
        print(f"  Note: dataset column 'reference_text' not found.")
        print(f"  Available columns: {ds.column_names}")
        print(f"  To compute precision/recall, provide ground-truth text from the")
        print(f"  Cochrane article or use the pre-downloaded labeled_facts/ data (Step 6).")
        return {}

    doi_to_gt_text  = {r["doi"]: r[gt_text_col]  for r in ds}
    doi_to_gt_facts = {r["doi"]: r[gt_facts_col] for r in ds} if gt_facts_col else {}

    label_results: dict[str, dict[str, dict]] = {}

    for config_label, entries in all_results.items():
        section(f"Config: {config_label}")
        doi_to_facts = config_to_doi_to_facts.get(config_label, {})
        config_results: dict[str, dict] = {}

        for entry in entries:
            doi        = entry["doi"]
            conclusion = entry["conclusion"]
            llm_facts  = doi_to_facts.get(doi, [])
            gt_text    = doi_to_gt_text.get(doi, "")
            gt_facts   = doi_to_gt_facts.get(doi, [])

            if not llm_facts:
                print(f"  [{doi}] No atomic facts — skipping.")
                continue
            if not gt_text:
                print(f"  [{doi}] No ground-truth text — skipping.")
                continue

            print(f"\n  Labeling: {doi}")
            print(f"  Model facts: {len(llm_facts)}  |  GT facts: {len(gt_facts) or '?'}  |  GT text: {len(gt_text)} chars")

            # ── Factual precision ─────────────────────────────────────────────
            # Each LLM atomic fact is judged against the ground-truth article text
            prec_result = precision_judge.compute_factual_precision(
                llm_atomic_facts  = llm_facts,
                ground_truth_text = gt_text,
            )

            # ── Factual recall ────────────────────────────────────────────────
            # Each ground-truth atomic fact is checked against the full model conclusion
            if gt_facts:
                rec_result = recall_judge.compute_factual_recall(
                    llm_response_text    = conclusion,
                    article_atomic_facts = gt_facts,
                )
            else:
                rec_result = {"factual_recall": None, "note": "No GT facts available"}

            p = prec_result.get("factual_precision")
            r = rec_result.get("factual_recall")
            f = compute_f1(p, r)

            print(f"  Factual Precision : {p:.3f}" if p is not None else "  Factual Precision : N/A")
            print(f"  Factual Recall    : {r:.3f}" if r is not None else "  Factual Recall    : N/A (no GT facts)")
            print(f"  Factual F1        : {f:.3f}" if f is not None else "  Factual F1        : N/A")

            config_results[doi] = {"precision": prec_result, "recall": rec_result}

        label_results[config_label] = config_results

    return label_results


# =============================================================================
# Step 6 — Compute aggregate metrics from pre-downloaded labeled facts
# =============================================================================

def step6_metrics(label_results: dict[str, dict[str, dict]]):
    """
    Computes macro-averaged Precision / Recall / F1 from the labeled results
    returned by step5_label.

    Metric formula (same as the paper):
      Precision = (S/T) × (1 – C/T)   per DOI, macro-averaged over DOIs
      Recall    = supported / total_gt  per DOI, macro-averaged over DOIs
      F1        = harmonic mean of macro Precision and macro Recall

    Results are printed as a table and returned as a dict.
    """
    banner("Step 6 · Compute Aggregate Metrics")

    from data_labeling.utils.precision_equations import compute_f1

    config_metrics: dict[str, dict] = {}

    for config_label, doi_results in label_results.items():
        section(f"Config: {config_label}")

        p_scores, r_scores = [], []

        for doi, scores in doi_results.items():
            p = scores.get("precision", {}).get("factual_precision")
            r = scores.get("recall",    {}).get("factual_recall")
            if p is not None:
                p_scores.append(p)
            if r is not None:
                r_scores.append(r)

        macro_p = sum(p_scores) / len(p_scores) if p_scores else 0.0
        macro_r = sum(r_scores) / len(r_scores) if r_scores else 0.0
        macro_f = compute_f1(macro_p, macro_r) or 0.0

        config_metrics[config_label] = {
            "precision": macro_p,
            "recall":    macro_r,
            "f1":        macro_f,
            "n_dois":    len(doi_results),
        }

    if not config_metrics:
        print("  No labeled results to aggregate.")
        return {}

    # ── Print results table ───────────────────────────────────────────────────
    C, P, R, F, N = 20, 30, 30, 30, 10
    print(f"\n  {'Config':<{C}} {'Factual Precision (Macro)':>{P}} {'Factual Recall (Macro)':>{R}} {'Factual F1 (Macro)':>{F}} {'N (DOIs)':>{N}}")
    print(f"  {'─'*C} {'─'*P} {'─'*R} {'─'*F} {'─'*N}")
    for name, m in sorted(config_metrics.items(), key=lambda x: -x[1]["f1"]):
        print(
            f"  {name:<{C}} {m['precision']:>{P}.3f} {m['recall']:>{R}.3f}"
            f" {m['f1']:>{F}.3f} {m['n_dois']:>{N}}"
        )

    return config_metrics


# =============================================================================
# CLI entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SciConBench end-to-end evaluation example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n", type=int, default=N_SAMPLES,
        help=f"Number of benchmark examples to query (default: {N_SAMPLES}).",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Model name (default: {MODEL!r}).",
    )
    parser.add_argument(
        "--provider", default=PROVIDER,
        help=f"LLM provider: openai | claude | gemini | perplexity (default: {PROVIDER!r}).",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Apply CLI overrides to module-level config
    global N_SAMPLES, MODEL, PROVIDER
    N_SAMPLES = args.n
    MODEL     = args.model
    PROVIDER  = args.provider

    ds            = step1_load_dataset()
    all_titles, doi_to_title, doi_to_date = step2_explore(ds)
    all_results   = await step3_query(ds, all_titles, doi_to_title, doi_to_date)
    doi_to_facts  = step4_atomic_facts(all_results)
    label_results = step5_label(ds, all_results, doi_to_facts)
    step6_metrics(label_results)

if __name__ == "__main__":
    asyncio.run(main())
