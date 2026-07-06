"""SQLAlchemy engine, session factory, and declarative base for SciConBench-Track.

All other ``db`` modules import ``Base``, ``Session``, and ``engine`` from here.
Call :func:`init_db` once at startup to create tables that do not yet exist.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

# Deferred engine/session — constructed in _setup() so that the db_path from
# config is only read after all modules have loaded.
engine = None
Session = None


def _setup(db_path: Path | None = None) -> None:
    """Initialize the module-level engine and Session factory."""
    global engine, Session

    if db_path is None:
        from config import path_cfg
        db_path = path_cfg.db_path

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Session = sessionmaker(bind=engine)
    logger.debug("DB engine configured: %s", db_path)


def init_db(db_path: Path | None = None, force: bool = False) -> None:
    """Create all tables that do not yet exist (or drop-and-recreate if *force*).

    When *db_path* is given, the module-level engine and Session are updated to
    point to that database for the remainder of the process.

    Safe to call multiple times — subsequent calls with ``force=False`` are no-ops
    if the tables already exist.
    """
    _setup(db_path)

    # Import models so their metadata is registered with Base before create_all.
    import db.db  # noqa: F401

    if force:
        Base.metadata.drop_all(engine)
        logger.warning("Dropped all tables (force=True).")

    Base.metadata.create_all(engine)
    logger.info("Database tables created/verified.")


# Auto-setup when the module is first imported so that ``Session`` is never None
# in normal usage.  If ``config`` isn't ready yet (e.g. during tests), the caller
# must invoke ``init_db(db_path=...)`` explicitly before using the session.
try:
    _setup()
except Exception:
    pass  # Will be retried when init_db() is called explicitly
