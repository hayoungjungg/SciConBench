"""Pipeline email notifications for SciConBench-Track.

Builds a running :class:`PipelineReport` during a workflow run, then emails a
plain-text digest on success or failure. Emails are best-effort: missing
credentials or SMTP errors never abort the pipeline.
"""

from __future__ import annotations

import os
import smtplib
import statistics
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from typing import Any, Iterable, Sequence


# ── Truncation helpers ─────────────────────────────────────────────────────────

_SAMPLE_N = 3
_EXCERPT = 400
_FACT_N = 5
_QUAL_N = 3


def _clip(text: str | None, n: int = _EXCERPT) -> str:
    s = " ".join((text or "").split())
    if len(s) <= n:
        return s
    return s[: n - 1].rstrip() + "…"


def _bullets(items: Iterable[str], n: int = _FACT_N, indent: str = "    ") -> str:
    lines = []
    for i, item in enumerate(items):
        if i >= n:
            lines.append(f"{indent}…")
            break
        lines.append(f"{indent}- {_clip(item, 220)}")
    return "\n".join(lines) if lines else f"{indent}(none)"


def _fmt(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


# ── Report ─────────────────────────────────────────────────────────────────────


@dataclass
class StageUpdate:
    name: str
    status: str  # "ok" | "failed" | "skipped"
    summary: str = ""
    detail: str = ""


@dataclass
class PipelineReport:
    """Accumulates stage progress + run metadata for the end-of-run email.

    When ``workflow_log`` is set, every stage begin/finish/fail and
    note/warn/error is also appended to the on-disk workflow log.
    """

    trial: bool = False
    target_month: str = ""
    max_dois: int | None = None
    providers: list[str] | None = None
    started_at: str = ""
    ended_at: str = ""
    current_stage: str | None = None
    stages: list[StageUpdate] = field(default_factory=list)
    new_dois: list[str] = field(default_factory=list)
    eval_dois: list[str] = field(default_factory=list)
    held_back: list[str] = field(default_factory=list)
    n_core: int = 0
    n_rolling: int = 0
    upload_note: str = ""
    extra_notes: list[str] = field(default_factory=list)
    workflow_log: Any = None  # optional WorkflowLog; avoid circular import type

    def begin(self, name: str) -> None:
        self.current_stage = name
        if self.workflow_log is not None:
            self.workflow_log.stage_begin(name)

    def finish(self, summary: str = "", detail: str = "") -> None:
        name = self.current_stage or "unknown"
        self.stages.append(StageUpdate(name=name, status="ok", summary=summary, detail=detail))
        if self.workflow_log is not None:
            self.workflow_log.stage_end(name, summary=summary, detail=detail)
        self.current_stage = None

    def fail_stage(self, summary: str = "", detail: str = "") -> None:
        name = self.current_stage or "unknown"
        self.stages.append(
            StageUpdate(name=name, status="failed", summary=summary, detail=detail)
        )
        if self.workflow_log is not None:
            self.workflow_log.stage_fail(name, summary=summary, detail=detail)
        self.current_stage = None

    def note(self, text: str) -> None:
        self.extra_notes.append(text)
        if self.workflow_log is not None:
            self.workflow_log.info(text)

    def warn(self, text: str) -> None:
        self.extra_notes.append(f"WARN: {text}")
        if self.workflow_log is not None:
            self.workflow_log.warn(text)

    def error(self, text: str) -> None:
        self.extra_notes.append(f"ERROR: {text}")
        if self.workflow_log is not None:
            self.workflow_log.error(text)

    # ── Email bodies ───────────────────────────────────────────────────────────

    def subject_success(self) -> str:
        tag = "TRIAL" if self.trial else "Track"
        return f"[SciConBench-{tag}] SUCCESS — {self.ended_at or self.started_at}"

    def subject_failure(self) -> str:
        tag = "TRIAL" if self.trial else "Track"
        stage = self.current_stage or (
            next((s.name for s in reversed(self.stages) if s.status == "failed"), None)
            or "unknown stage"
        )
        return f"[SciConBench-{tag}] FAILED @ {stage} — {self.started_at}"

    def render_success(self) -> str:
        parts = [
            self._header("SUCCESS"),
            self._config_block(),
            self._stage_log(),
            self._panels_block(),
            self._discovery_block(),
            self._sample_digest(),
            self._metrics_block(),
        ]
        if self.extra_notes:
            parts.append("NOTES\n" + "\n".join(f"  • {n}" for n in self.extra_notes))
        parts.append(
            f"\nFinished: {self.ended_at}\n"
            "This is an automated SciConBench-Track notification."
        )
        return "\n\n".join(p for p in parts if p)

    def render_failure(self, error: str) -> str:
        stage = self.current_stage or "unknown (before stage tracking / between stages)"
        parts = [
            self._header("FAILED"),
            self._config_block(),
            f"FAILED AT STAGE\n  {stage}",
            self._stage_log(),
            self._panels_block(),
            self._discovery_block(),
            "ERROR\n" + _indent(error.rstrip(), "  "),
            "Partial sample digest (whatever completed before the failure):",
            self._sample_digest(),
        ]
        parts.append(
            "\nThis is an automated SciConBench-Track notification. "
            "The pipeline raised after the error above."
        )
        return "\n\n".join(p for p in parts if p)

    # ── Sections ───────────────────────────────────────────────────────────────

    def _header(self, status: str) -> str:
        mode = "trial" if self.trial else "production"
        return (
            f"SciConBench-Track — {status}\n"
            f"{'=' * 72}\n"
            f"Mode: {mode}   Closed month for evals: {self.target_month}\n"
            f"Started: {self.started_at}"
        )

    def _config_block(self) -> str:
        providers = ", ".join(self.providers) if self.providers else "all configured"
        lines = [
            "RUN CONFIG",
            f"  max_dois:     {self.max_dois if self.max_dois is not None else 'none (uncapped)'}",
            f"  providers:    {providers}",
            f"  core panel:   {self.n_core} DOI(s)",
            f"  rolling:      {self.n_rolling} DOI(s) total",
            f"  eval universe:{len(self.eval_dois)} DOI(s)"
            + (f"  (held back: {len(self.held_back)})" if self.held_back else ""),
        ]
        if self.upload_note:
            lines.append(f"  upload:       {self.upload_note}")
        if self.workflow_log is not None and getattr(self.workflow_log, "path", None):
            lines.append(f"  workflow log: {self.workflow_log.path}")
        return "\n".join(lines)

    def _stage_log(self) -> str:
        if not self.stages and not self.current_stage:
            return "STAGE LOG\n  (no stages recorded)"
        lines = ["STAGE LOG"]
        for s in self.stages:
            mark = {"ok": "✓", "failed": "✗", "skipped": "·"}.get(s.status, "?")
            line = f"  {mark} {s.name}"
            if s.summary:
                line += f" — {s.summary}"
            lines.append(line)
            if s.detail:
                for dline in s.detail.strip().splitlines():
                    lines.append(f"      {dline}")
        if self.current_stage:
            lines.append(f"  … in progress / failed here: {self.current_stage}")
        return "\n".join(lines)

    def _panels_block(self) -> str:
        """Counts per rolling cohort month, plus whether that panel is evaluated."""
        closed = self.target_month or ""
        lines = [
            "ROLLING PANELS BY COHORT MONTH",
            f"  Closed month for evals: {closed or '(unset)'}",
            f"  Rule: cohort_month <= closed month → eligible to RUN; "
            f"later / unassigned → HELD BACK (ingested, not queried yet).",
        ]
        if self.trial:
            lines.append(
                "  Trial note: this run only queries newly discovered DOIs that are "
                "also eligible; other eligible rolling DOIs are skipped."
            )
        lines.append("")
        lines.append(
            f"  {'cohort':<14} {'n':>4}  {'status':<12}  note"
        )
        lines.append(f"  {'-' * 14} {'-' * 4}  {'-' * 12}  {'-' * 40}")

        try:
            rows = _rolling_cohort_counts(closed)
        except Exception as exc:
            lines.append(f"  (could not load panel counts: {exc})")
            return "\n".join(lines)

        if not rows:
            lines.append("  (no rolling DOIs in the database yet)")
        else:
            total = 0
            n_run = 0
            n_hold = 0
            for cohort, n, status, note in rows:
                total += n
                if status == "RUN":
                    n_run += n
                else:
                    n_hold += n
                # Mark RUN vs HELD BACK relative to closed month. For trial,
                # "RUN" means the cohort is eligible; actual query set may be
                # a subset (new_dois only) — clarified above.
                lines.append(
                    f"  {cohort:<14} {n:>4}  {status:<12}  {note}"
                )
            lines.append("")
            lines.append(
                f"  Rolling total: {total}  "
                f"(eligible to run: {n_run}; held back: {n_hold})"
            )
            self.n_rolling = total

        lines.append("")
        lines.append(
            f"  Core panel: {self.n_core} DOI(s) — always eligible when "
            f"questions+facts are ready (not cohort-gated)."
        )
        return "\n".join(lines)

    def _discovery_block(self) -> str:
        """DOI list only for newly discovered rolling reviews this run."""
        if not self.new_dois:
            return (
                "NEW ROLLING DOIs THIS RUN\n"
                "  No new rolling DOI(s) registered this run."
            )
        lines = [
            f"NEW ROLLING DOIs THIS RUN ({len(self.new_dois)} added)",
            "  (titles/cohorts filled in after text extraction when available)",
        ]
        try:
            from db.utils import get_doi_info_rows, get_reviews_from_db
            reviews = get_reviews_from_db()
            info = {r["doi"]: r for r in get_doi_info_rows()}
        except Exception:
            reviews = {}
            info = {}

        closed = self.target_month or ""
        for doi in self.new_dois:
            title = (reviews.get(doi) or {}).get("name") or ""
            cohort = (info.get(doi) or {}).get("cohort_month")
            if cohort and closed and cohort <= closed:
                gate = "eligible to RUN"
            elif cohort:
                gate = "HELD BACK (cohort still open)"
            else:
                gate = "HELD BACK (cohort not assigned yet)"
            bit = f"  • {doi}"
            if cohort:
                bit += f"  [{cohort}]"
            bit += f"  — {gate}"
            if title:
                bit += f"\n      {_clip(title, 100)}"
            lines.append(bit)
        return "\n".join(lines)

    def _sample_digest(self) -> str:
        """Build qualitative samples for up to ``_SAMPLE_N`` eval (or new) DOIs."""
        dois = list(self.eval_dois or self.new_dois)[:_SAMPLE_N]
        if not dois:
            return "SAMPLE DIGEST\n  (no DOIs to sample)"
        try:
            return "SAMPLE DIGEST\n" + build_sample_digest(
                dois, run_month=self.target_month or None
            )
        except Exception as exc:
            return f"SAMPLE DIGEST\n  (failed to build digest: {exc})"

    def _metrics_block(self) -> str:
        try:
            from db.utils import get_eval_metrics
            rows = get_eval_metrics(
                run_month=self.target_month or None,
                dois=self.eval_dois or None,
            )
        except Exception as exc:
            return f"MACRO METRICS\n  (unavailable: {exc})"
        if not rows:
            return "MACRO METRICS\n  (no scored responses)"

        by_model: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            by_model.setdefault(r["model"], []).append(r)

        lines = [
            "MACRO METRICS BY MODEL (per-DOI average)",
            f"  {'model':<28} {'n':>3}  {'P':>6}  {'R':>6}  {'F1':>6}",
            f"  {'-' * 28} {'-' * 3}  {'-' * 6}  {'-' * 6}  {'-' * 6}",
        ]
        for model, group in sorted(by_model.items()):
            ps = [g["precision"] for g in group if g["precision"] is not None]
            rs = [g["recall"] for g in group if g["recall"] is not None]
            f1s = [g["f1"] for g in group if g["f1"] is not None]
            lines.append(
                f"  {model:<28} {len(group):>3}  "
                f"{_fmt(statistics.mean(ps) if ps else None):>6}  "
                f"{_fmt(statistics.mean(rs) if rs else None):>6}  "
                f"{_fmt(statistics.mean(f1s) if f1s else None):>6}"
            )

        # Per-DOI table (compact)
        lines.append("")
        lines.append("PER-DOI SCORES")
        lines.append(f"  {'doi':<42} {'model':<18} {'P':>6} {'R':>6} {'F1':>6}")
        for r in sorted(rows, key=lambda x: (x["doi"], x["model"])):
            lines.append(
                f"  {r['doi']:<42} {r['model']:<18} "
                f"{_fmt(r['precision']):>6} {_fmt(r['recall']):>6} {_fmt(r['f1']):>6}"
            )
        return "\n".join(lines)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _rolling_cohort_counts(
    closed_month: str,
) -> list[tuple[str, int, str, str]]:
    """Return ``[(cohort_label, n, status, note), ...]`` for rolling DOIs.

    ``status`` is ``RUN`` when ``cohort <= closed_month``, else ``HELD BACK``.
    Unassigned cohorts (no publication date yet) are always held back.
    Sorted by cohort month ascending; unassigned last.
    """
    from collections import Counter

    from db.db import PanelType
    from db.utils import get_doi_info_rows

    counts: Counter[str] = Counter()
    for row in get_doi_info_rows():
        if row.get("panel_type") != PanelType.ROLLING.value:
            continue
        cohort = row.get("cohort_month") or ""
        key = cohort if cohort else "(unassigned)"
        counts[key] += 1

    def _sort_key(label: str) -> tuple[int, str]:
        if label == "(unassigned)":
            return (1, "")
        return (0, label)

    out: list[tuple[str, int, str, str]] = []
    for label in sorted(counts, key=_sort_key):
        n = counts[label]
        if label == "(unassigned)":
            out.append((
                label, n, "HELD BACK",
                "no publication date / cohort not set yet",
            ))
        elif closed_month and label <= closed_month:
            out.append((
                label, n, "RUN",
                f"cohort <= {closed_month} (eligible this run)",
            ))
        else:
            out.append((
                label, n, "HELD BACK",
                f"cohort > {closed_month or '?'} (still-open month)",
            ))
    return out


# ── Sample digest from DB ──────────────────────────────────────────────────────


def build_sample_digest(
    dois: Sequence[str],
    *,
    run_month: str | None = None,
) -> str:
    """Plain-text qualitative samples for the given DOIs (questions, facts, evals)."""
    from db.utils import (
        get_all_model_responses,
        get_atomic_facts,
        get_questions,
        get_reviews_from_db,
    )

    reviews = get_reviews_from_db()
    questions = get_questions()
    cochrane = get_atomic_facts("cochrane")
    responses = get_all_model_responses(run_month=run_month)

    # Group responses by DOI
    by_doi: dict[str, list[dict[str, Any]]] = {}
    for rid, data in responses.items():
        if data.get("doi") not in dois:
            continue
        entry = dict(data)
        entry["_response_id"] = rid
        by_doi.setdefault(data["doi"], []).append(entry)

    blocks: list[str] = []
    for i, doi in enumerate(dois, 1):
        rev = reviews.get(doi) or {}
        q = questions.get(doi) or ""
        facts = (cochrane.get(doi) or {}).get("all_facts") or []
        objs = rev.get("objectives") or ""
        conclusions = rev.get("authors_conclusions") or ""
        ref = rev.get("reference_text") or ""

        lines = [
            f"── Sample {i}/{len(dois)} ──",
            f"DOI:   {doi}",
            f"Title: {_clip(rev.get('name'), 160)}",
            f"Pub:   {rev.get('publication_date') or '—'}",
            "",
            "Reference text (excerpt):",
            f"  {_clip(ref, 500)}",
            "",
            "BEFORE question generation — objectives/background:",
            f"  {_clip(objs or ref, 400)}",
            "AFTER — generated clinical question:",
            f"  {_clip(q, 500) or '(none)'}",
            "",
            "BEFORE Cochrane atomic facts — authors' conclusions:",
            f"  {_clip(conclusions or ref, 400)}",
            f"AFTER — {len(facts)} Cochrane atomic fact(s):",
            _bullets(facts, _FACT_N),
        ]

        resp_list = by_doi.get(doi) or []
        if not resp_list:
            lines.append("")
            lines.append("Model responses: (none for this run_month)")
        else:
            lines.append("")
            lines.append(f"Model responses ({len(resp_list)}):")
            for resp in resp_list:
                lines.extend(_format_response_block(resp))

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _facts_from_pairs(pairs) -> list[str]:
    out: list[str] = []
    for p in pairs or []:
        if isinstance(p, dict):
            out.extend(str(x) for x in (p.get("atomic_facts") or []))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            out.extend(str(x) for x in (p[1] or []))
    return out


def _format_response_block(resp: dict[str, Any]) -> list[str]:
    """One model response: conclusion excerpt, facts, P/R/F1, qualitative labels."""
    import re

    from db import Session as _Session
    from db.db import AtomicFacts, AtomicFactSource, FactualPrecisionResult, FactualRecallResult

    rid = resp.get("_response_id")
    model = resp.get("model") or "?"
    provider = resp.get("provider") or "?"
    raw = resp.get("response") or ""
    m = re.search(r"\[\[\[(.*?)\]\]\]", raw, re.DOTALL)
    conclusion = (m.group(1).strip() if m else raw) or ""

    lines = [
        f"  • {provider}/{model}",
        f"    Conclusion excerpt: {_clip(conclusion, 350)}",
    ]

    try:
        with _Session()() as session:
            af = (
                session.query(AtomicFacts)
                .filter(
                    AtomicFacts.id == rid,
                    AtomicFacts.source == AtomicFactSource.MODEL_RESPONSE,
                )
                .first()
            )
            prec = (
                session.query(FactualPrecisionResult)
                .filter(FactualPrecisionResult.model_response_id == rid)
                .first()
            )
            rec = (
                session.query(FactualRecallResult)
                .filter(FactualRecallResult.model_response_id == rid)
                .first()
            )
    except Exception as exc:
        lines.append(f"    (could not load grades/facts: {exc})")
        return lines

    if af is not None:
        all_facts = _facts_from_pairs(af.atomic_facts_pairs or [])
        lines.append(f"    Response atomic facts ({len(all_facts)}), sample:")
        lines.append(_bullets(all_facts, _FACT_N, indent="      "))
    else:
        lines.append("    Response atomic facts: (none yet)")

    if prec is not None and rec is not None:
        p, r = prec.factual_precision, rec.factual_recall
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        lines.append(
            f"    Scores: P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
            f"(LLM facts {prec.supported_facts}/{prec.total_llm_facts} supported, "
            f"{prec.contradicted_facts} contradicted; "
            f"article facts {rec.supported_facts}/{rec.total_article_facts} covered)"
        )
        contradicted = list(prec.contradicted_facts_list or [])
        if contradicted:
            lines.append("    Qualitative — CONTRADICTED response facts:")
            lines.append(_bullets(contradicted, _QUAL_N, indent="      "))
        not_sup = list(prec.not_supported_facts_list or [])
        if not_sup:
            lines.append("    Qualitative — NOT SUPPORTED response facts:")
            lines.append(_bullets(not_sup, _QUAL_N, indent="      "))
        missed = list(rec.not_supported_facts_list or [])
        if missed:
            lines.append("    Qualitative — Cochrane facts MISSED (recall gaps):")
            lines.append(_bullets(missed, _QUAL_N, indent="      "))
    else:
        lines.append("    Scores: (precision/recall not yet graded)")

    return lines


# ── SMTP ───────────────────────────────────────────────────────────────────────


def send_email(subject: str, body: str) -> None:
    """Send a plain-text email. Never raises — logs and continues on failure."""
    try:
        sender = os.environ["EMAIL_SENDER"]
        password = os.environ["EMAIL_APP_PASSWORD"]
        recipient = os.environ["EMAIL_RECIPIENT"]
    except KeyError as missing:
        print(f"Email notification skipped (missing {missing}): {subject}")
        return

    try:
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


def notify_start(report: PipelineReport) -> None:
    mode = "TRIAL" if report.trial else "Track"
    providers = ", ".join(report.providers) if report.providers else "all configured"
    body = (
        f"SciConBench-Track pipeline started.\n\n"
        f"Mode:          {'trial' if report.trial else 'production'}\n"
        f"Closed month:  {report.target_month}\n"
        f"max_dois:      {report.max_dois if report.max_dois is not None else 'none'}\n"
        f"providers:     {providers}\n"
        f"Started:       {report.started_at}\n\n"
        f"You will get another email when the run finishes (success or failure)."
    )
    send_email(f"[SciConBench-{mode}] Started — {report.started_at}", body)


def notify_success(report: PipelineReport) -> None:
    send_email(report.subject_success(), report.render_success())


def notify_failure(report: PipelineReport, error: str) -> None:
    send_email(report.subject_failure(), report.render_failure(error))
