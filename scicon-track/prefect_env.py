"""Point Prefect's local SQLite at a local disk before Prefect is imported.

Prefect 3 stores orchestration / telemetry state in ``$PREFECT_HOME/prefect.db``.
On NFS (e.g. ``/n/fs/hamcore/...``) SQLite locking fails with
``OperationalError: database is locked`` — often on a ``TELEMETRY_SESSION``
insert during flow startup. This helper must run *before*
``from prefect import ...``.

Priority:
  1. Explicit ``PREFECT_HOME`` env var (also loadable from ``.env``)
  2. If the default ``~/.prefect`` would land on NFS, use
     ``/tmp/<user>-prefect`` instead
  3. Otherwise leave Prefect's default alone
"""

from __future__ import annotations

import os
from pathlib import Path


def _is_nfs(path: Path) -> bool:
    """True if *path* resolves onto an NFS (or similar network) mount."""
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        resolved = path.expanduser()

    # Fast path for Princeton /n/fs/* project stores.
    if str(resolved).startswith("/n/fs/"):
        return True

    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            mounts = []
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    mounts.append((parts[1], parts[2]))  # (mountpoint, fstype)
    except OSError:
        return False

    mounts.sort(key=lambda m: len(m[0]), reverse=True)
    resolved_s = str(resolved)
    for mountpoint, fstype in mounts:
        if resolved_s == mountpoint or resolved_s.startswith(
            mountpoint.rstrip("/") + "/"
        ):
            return fstype.startswith("nfs") or fstype in {
                "lustre",
                "gpfs",
                "fuse.sshfs",
            }
    return False


def configure_prefect_home() -> Path:
    """Ensure ``PREFECT_HOME`` points at a writable local directory. Idempotent.

    Also raises Prefect's ephemeral-server startup budget (used by
    ``.serve()`` / scheduled runs) so cold starts on shared clusters are
    less likely to hit ``Timed out while attempting to connect…``.
    ``--once`` bypasses the ephemeral server entirely via ``flow.fn()``.
    """
    # Give the ephemeral server more than the default 20s when it *is* used.
    os.environ.setdefault("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "120")
    os.environ.setdefault("PREFECT_LOGGING_TO_API_WHEN_MISSING_FLOW", "ignore")

    existing = os.environ.get("PREFECT_HOME")
    if existing:
        path = Path(existing).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    try:
        from dotenv import load_dotenv

        repo_env = Path(__file__).resolve().parent.parent / ".env"
        load_dotenv(repo_env, override=False)
        load_dotenv(override=False)
    except Exception:
        pass

    existing = os.environ.get("PREFECT_HOME")
    if existing:
        path = Path(existing).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    default = Path.home() / ".prefect"
    if _is_nfs(default) or _is_nfs(Path.home()):
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "scicon"
        path = Path("/tmp") / f"{user}-prefect"
        os.environ["PREFECT_HOME"] = str(path)
        path.mkdir(parents=True, exist_ok=True)
        print(
            f"Prefect: PREFECT_HOME → {path} "
            f"(default {default} is on NFS; SQLite locks fail there)."
        )
        return path

    default.mkdir(parents=True, exist_ok=True)
    return default
