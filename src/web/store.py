"""
SQLite persistence for Pulse run history.

One row per pipeline run, carrying it through the whole lifecycle: queued ->
running -> awaiting_approval -> delivering -> delivered (or rejected/failed).

The generated report is stored in the row rather than left on disk, so an
approval survives a process restart. Only the PDF itself stays on the
filesystem, which is why the reports directory needs a persistent volume.

A connection is opened per operation rather than shared, since the pipeline
runs on a background thread while HTTP requests are served on others.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Lifecycle states. Terminal states are those a run can never leave.
QUEUED = "queued"
RUNNING = "running"
AWAITING_APPROVAL = "awaiting_approval"
DELIVERING = "delivering"
DELIVERED = "delivered"
REJECTED = "rejected"
FAILED = "failed"

TERMINAL_STATES = {DELIVERED, REJECTED, FAILED}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    product           TEXT NOT NULL,
    week_start        TEXT NOT NULL,
    week_end          TEXT NOT NULL,
    status            TEXT NOT NULL,
    step              TEXT,
    error             TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    review_count      INTEGER DEFAULT 0,
    theme_count       INTEGER DEFAULT 0,
    validation_passed INTEGER DEFAULT 0,
    validation_errors TEXT,
    themes            TEXT,
    report_markdown   TEXT,
    fee_explainer     TEXT,
    fee_pain_point    TEXT,
    pdf_path          TEXT,
    pdf_url           TEXT,
    email_sent        INTEGER DEFAULT 0,
    recipients        TEXT,
    approved_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs (created_at DESC);
"""

# JSON-encoded columns, decoded transparently on read.
_JSON_COLUMNS = ("validation_errors", "themes", "recipients")


def _now() -> str:
    """Current UTC timestamp, second resolution, ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> str:
    """Where the database lives.

    Defaults inside the data directory so a single Railway volume mount
    covers the database, the PDFs and the embedding cache together.
    """
    from src.config import DATA_DIR

    return os.getenv("PULSE_DB_PATH", os.path.join(DATA_DIR, "pulse.db"))


@contextmanager
def _connect():
    """Yield a connection with row access by name, committing on success."""
    path = db_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        # WAL lets the HTTP thread read while the pipeline thread writes.
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema if it does not exist."""
    with _connect() as conn:
        conn.executescript(_SCHEMA)
    logger.info("Pulse database ready at %s", db_path())


def _decode(row: sqlite3.Row) -> dict:
    """Turn a row into a dict, decoding JSON columns and booleans."""
    run = dict(row)
    for column in _JSON_COLUMNS:
        raw = run.get(column)
        if raw:
            try:
                run[column] = json.loads(raw)
            except json.JSONDecodeError:
                run[column] = []
        else:
            run[column] = []
    run["validation_passed"] = bool(run.get("validation_passed"))
    run["email_sent"] = bool(run.get("email_sent"))
    return run


def create_run(product: str, week_start: str, week_end: str) -> str:
    """Insert a queued run and return its id."""
    run_id = uuid.uuid4().hex[:12]
    now = _now()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, product, week_start, week_end, status, "
            "step, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, product, week_start, week_end, QUEUED, "Queued", now, now),
        )

    logger.info("Run %s created: %s %s to %s", run_id, product, week_start, week_end)
    return run_id


def update_run(run_id: str, **fields) -> None:
    """Update the given columns on a run, always touching ``updated_at``.

    JSON columns accept Python lists directly.
    """
    if not fields:
        return

    for column in _JSON_COLUMNS:
        if column in fields and not isinstance(fields[column], (str, type(None))):
            fields[column] = json.dumps(fields[column])

    for column in ("validation_passed", "email_sent"):
        if column in fields:
            fields[column] = int(bool(fields[column]))

    fields["updated_at"] = _now()
    assignments = ", ".join(f"{k} = ?" for k in fields)

    with _connect() as conn:
        conn.execute(
            f"UPDATE runs SET {assignments} WHERE id = ?",
            (*fields.values(), run_id),
        )


def get_run(run_id: str) -> dict | None:
    """Fetch one run, or None if the id is unknown."""
    with _connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _decode(row) if row else None


def list_runs(limit: int = 50) -> list[dict]:
    """Fetch recent runs, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_decode(r) for r in rows]


def active_run() -> dict | None:
    """The run currently occupying the pipeline, if any.

    Used to refuse a second concurrent trigger: the pipeline writes to shared
    paths under the reports directory, so two at once would collide.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM runs WHERE status IN (?, ?, ?) "
            "ORDER BY created_at DESC LIMIT 1",
            (QUEUED, RUNNING, DELIVERING),
        ).fetchone()
    return _decode(row) if row else None


def reset_stuck_runs() -> int:
    """Mark interrupted runs as failed on startup.

    A run in a non-terminal, non-approval state cannot resume after a restart
    because its work happened in a thread that no longer exists. Runs awaiting
    approval are untouched — their results live in the database and are still
    deliverable.
    """
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE runs SET status = ?, error = ?, updated_at = ? "
            "WHERE status IN (?, ?, ?)",
            (
                FAILED,
                "Interrupted by a server restart.",
                _now(),
                QUEUED,
                RUNNING,
                DELIVERING,
            ),
        )
        count = cursor.rowcount

    if count:
        logger.warning("Marked %d interrupted run(s) as failed.", count)
    return count
