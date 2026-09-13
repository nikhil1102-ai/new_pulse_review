"""
Pulse — the web UI for triggering, reviewing and approving review-pulse runs.

A small FastAPI app over :mod:`src.web.runner`. The browser polls the JSON
endpoints; there is no build step and no frontend framework.

Endpoints:
    GET  /                       the single-page UI
    GET  /api/runs               run history
    POST /api/runs               trigger a run
    GET  /api/runs/{id}          one run, including the report for review
    GET  /api/runs/{id}/pdf      download the generated PDF
    POST /api/runs/{id}/approve  approve -> upload PDF and draft the email
    POST /api/runs/{id}/reject   reject -> nothing is ever sent
    GET  /api/health             liveness probe

Run:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from src.config import REVIEW_WINDOW_WEEKS
from src.web import runner, store

logger = logging.getLogger(__name__)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

@asynccontextmanager
async def _lifespan(_: FastAPI):
    """Prepare the database and clear runs orphaned by a restart."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    store.init_db()
    store.reset_stuck_runs()
    yield


app = FastAPI(
    title="Pulse",
    description="Trigger, review and approve weekly review-pulse reports",
    version="1.0.0",
    lifespan=_lifespan,
)


# ── Request models ───────────────────────────────────────────


class TriggerRequest(BaseModel):
    """Body for POST /api/runs. Every field is optional."""

    product: str | None = None
    week_start: str | None = None
    week_end: str | None = None


# ── Helpers ──────────────────────────────────────────────────


def _resolve_window(req: TriggerRequest) -> tuple[str, str, str]:
    """Fill in the product and date window, defaulting to the last 8 weeks."""
    product = (req.product or "Groww").strip() or "Groww"

    today = datetime.now(timezone.utc).date()
    try:
        week_end = (
            datetime.strptime(req.week_end, "%Y-%m-%d").date()
            if req.week_end
            else today
        )
        week_start = (
            datetime.strptime(req.week_start, "%Y-%m-%d").date()
            if req.week_start
            else week_end - timedelta(weeks=REVIEW_WINDOW_WEEKS)
        )
    except ValueError as exc:
        raise HTTPException(400, f"Dates must be YYYY-MM-DD: {exc}")

    if week_start >= week_end:
        raise HTTPException(400, "week_start must be earlier than week_end.")

    return product, week_start.isoformat(), week_end.isoformat()


def _require_run(run_id: str) -> dict:
    """Fetch a run or raise 404."""
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"No run with id {run_id}.")
    return run


# ── UI ───────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Serve the single-page UI."""
    path = os.path.join(STATIC_DIR, "index.html")
    with open(path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ── API ──────────────────────────────────────────────────────


@app.get("/api/health")
def health() -> dict:
    """Liveness probe, also reporting whether the pipeline is busy."""
    active = store.active_run()
    return {
        "status": "ok",
        "busy": active is not None,
        "active_run": active["id"] if active else None,
    }


@app.get("/api/runs")
def list_runs(limit: int = 50) -> dict:
    """Run history, newest first, without the bulky report bodies."""
    runs = store.list_runs(limit=min(max(limit, 1), 200))
    for run in runs:
        run.pop("report_markdown", None)
        run.pop("fee_explainer", None)
        run.pop("themes", None)
    return {"runs": runs}


@app.post("/api/runs", status_code=202)
def trigger_run(req: TriggerRequest) -> dict:
    """Start a pipeline run in the background.

    Returns 202 immediately; the run takes minutes. Returns 409 when another
    run is already in progress, since runs share paths under the reports
    directory and would collide.
    """
    product, week_start, week_end = _resolve_window(req)

    try:
        run_id = runner.start_run(product, week_start, week_end)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))

    return {
        "run_id": run_id,
        "product": product,
        "week_start": week_start,
        "week_end": week_end,
        "status": store.QUEUED,
    }


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """One run in full, including the report text for review."""
    return _require_run(run_id)


@app.get("/api/runs/{run_id}/pdf")
def download_pdf(run_id: str) -> FileResponse:
    """Download the generated PDF so it can be checked before approving."""
    run = _require_run(run_id)
    path = run.get("pdf_path")

    if not path or not os.path.exists(path):
        raise HTTPException(404, "No PDF was produced for this run.")

    return FileResponse(
        path, media_type="application/pdf", filename=os.path.basename(path)
    )


@app.post("/api/runs/{run_id}/approve")
def approve_run(run_id: str) -> dict:
    """Approve a reviewed run: upload the PDF and draft the email.

    This is the point at which anything leaves the machine.
    """
    try:
        runner.approve(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))

    logger.info("Run %s approved for delivery.", run_id)
    return {"run_id": run_id, "status": store.DELIVERING}


@app.post("/api/runs/{run_id}/reject")
def reject_run(run_id: str) -> dict:
    """Reject a reviewed run. Nothing is sent, and it cannot be approved later."""
    try:
        runner.reject(run_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))

    return {"run_id": run_id, "status": store.REJECTED}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
    )
