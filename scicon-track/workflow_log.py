"""Stage-level workflow log for SciConBench-Track.

Writes a single plain-text file under ``data_track/logs/`` summarizing how
the pipeline progressed (counts per stage, warnings, errors). It does *not*
dump per-DOI model responses — those live in ``data_track/results/``.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO
from zoneinfo import ZoneInfo

_TZ = ZoneInfo("America/New_York")
_lock = threading.Lock()
_active: "WorkflowLog | None" = None


def get_workflow_log() -> "WorkflowLog | None":
    return _active


def set_workflow_log(log: "WorkflowLog | None") -> None:
    global _active
    _active = log


def _stamp() -> str:
    return datetime.now(_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


class WorkflowLog:
    """Append-only stage progress log for one pipeline run."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")
        set_workflow_log(self)

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open_run(self, **meta: object) -> None:
        self._write("=" * 72)
        self._write(f"SciConBench-Track workflow log  started {_stamp()}")
        self._write(f"log file: {self.path}")
        for key, val in meta.items():
            self._write(f"  {key}: {val}")
        self._write("-" * 72)
        self._flush()

    def close_run(self, *, status: str, ended_at: str = "") -> None:
        self._write("-" * 72)
        self._write(
            f"RUN {status.upper()}  ended {ended_at or _stamp()}"
        )
        self._write("=" * 72)
        self._flush()
        try:
            self._fh.close()
        except Exception:
            pass
        if get_workflow_log() is self:
            set_workflow_log(None)

    # ── stage / messages ───────────────────────────────────────────────────────

    def stage_begin(self, name: str) -> None:
        self._write(f"[{_stamp()}] >> BEGIN  {name}")
        self._flush()

    def stage_end(self, name: str, summary: str = "", detail: str = "") -> None:
        line = f"[{_stamp()}] << OK     {name}"
        if summary:
            line += f" — {summary}"
        self._write(line)
        if detail:
            for dline in detail.strip().splitlines():
                self._write(f"           {dline}")
        self._flush()

    def stage_fail(self, name: str, summary: str = "", detail: str = "") -> None:
        line = f"[{_stamp()}] !! FAIL  {name}"
        if summary:
            line += f" — {summary}"
        self._write(line)
        if detail:
            for dline in detail.strip().splitlines():
                self._write(f"           {dline}")
        self._flush()

    def info(self, msg: str) -> None:
        self._write(f"[{_stamp()}]    INFO  {msg}")
        self._flush()

    def warn(self, msg: str) -> None:
        self._write(f"[{_stamp()}]    WARN  {msg}")
        self._flush()

    def error(self, msg: str) -> None:
        self._write(f"[{_stamp()}]    ERROR {msg}")
        self._flush()

    def exception(self, traceback_text: str) -> None:
        self._write(f"[{_stamp()}]    ERROR traceback:")
        for line in traceback_text.rstrip().splitlines():
            self._write(f"           {line}")
        self._flush()

    # ── internals ──────────────────────────────────────────────────────────────

    def _write(self, line: str) -> None:
        with _lock:
            try:
                self._fh.write(line.rstrip() + "\n")
            except Exception:
                pass

    def _flush(self) -> None:
        with _lock:
            try:
                self._fh.flush()
            except Exception:
                pass


def create_workflow_log(*, trial: bool = False, target_month: str = "") -> WorkflowLog:
    """Create a timestamped log under ``data_track/logs/``."""
    from config import path_cfg

    logs_dir = path_cfg.logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(_TZ).strftime("%Y%m%d-%H%M%S")
    mode = "trial" if trial else "prod"
    month = target_month.replace("-", "") if target_month else "na"
    path = logs_dir / f"workflow-{mode}-{month}-{ts}.log"
    return WorkflowLog(path)
