"""
Unit tests for Pulse — the trigger / review / approve web UI.

Covers src/web/store.py, src/web/runner.py and src/web/app.py. The pipeline
and the delivery node are mocked throughout: these tests verify the state
machine and the API contract, not the pipeline itself.

The critical property under test is that nothing reaches Google until a run
has been explicitly approved.
"""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.web import runner, store


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Point the store at a throwaway database for each test."""
    path = tmp_path / "pulse.db"
    monkeypatch.setenv("PULSE_DB_PATH", str(path))
    store.init_db()
    return path


@pytest.fixture
def client(db):
    """A test client whose startup hook reuses the temporary database."""
    from src.web.app import app

    with TestClient(app) as c:
        yield c


def _finished_run(**overrides) -> str:
    """Create a run already parked at awaiting_approval."""
    run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
    fields = {
        "status": store.AWAITING_APPROVAL,
        "step": "Ready for review",
        "review_count": 4821,
        "theme_count": 6,
        "validation_passed": True,
        "validation_errors": [],
        "themes": [{"label": "Crashes", "count": 1200, "share_pct": 40.0,
                    "avg_rating": 2.1, "quotes": ["it crashes"]}],
        "report_markdown": "## Summary\nUsers report crashes.",
        "fee_explainer": "- Brokerage is zero.",
        "pdf_path": "data/reports/fake.pdf",
    }
    fields.update(overrides)
    store.update_run(run_id, **fields)
    return run_id


# ──────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────


class TestStore:
    def test_create_and_fetch(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
        run = store.get_run(run_id)

        assert run["status"] == store.QUEUED
        assert run["product"] == "Groww"
        assert run["created_at"]

    def test_unknown_id_returns_none(self, db):
        assert store.get_run("nope") is None

    def test_update_roundtrips_json_and_bools(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
        store.update_run(
            run_id,
            validation_errors=["a", "b"],
            themes=[{"label": "x", "count": 1}],
            validation_passed=True,
            email_sent=False,
        )
        run = store.get_run(run_id)

        assert run["validation_errors"] == ["a", "b"]
        assert run["themes"][0]["label"] == "x"
        assert run["validation_passed"] is True
        assert run["email_sent"] is False

    def test_missing_json_columns_decode_to_lists(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
        run = store.get_run(run_id)
        assert run["validation_errors"] == [] and run["themes"] == []

    def test_list_runs_is_newest_first(self, db):
        first = store.create_run("A", "2026-08-01", "2026-08-08")
        second = store.create_run("B", "2026-08-01", "2026-08-08")
        assert [r["id"] for r in store.list_runs()][:2] == [second, first]

    def test_active_run_detects_in_flight_work(self, db):
        assert store.active_run() is None
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
        assert store.active_run()["id"] == run_id

        store.update_run(run_id, status=store.DELIVERED)
        assert store.active_run() is None

    def test_awaiting_approval_is_not_active(self, db):
        """An unapproved run must not block the next trigger forever."""
        _finished_run()
        assert store.active_run() is None

    def test_reset_stuck_runs_spares_pending_approvals(self, db):
        running = store.create_run("Groww", "2026-08-10", "2026-09-07")
        store.update_run(running, status=store.RUNNING)
        waiting = _finished_run()

        assert store.reset_stuck_runs() == 1
        assert store.get_run(running)["status"] == store.FAILED
        assert store.get_run(waiting)["status"] == store.AWAITING_APPROVAL


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────


class TestRunner:
    def test_refuses_concurrent_runs(self, db):
        with patch("src.web.runner._spawn"):
            runner.start_run("Groww", "2026-08-10", "2026-09-07")
            with pytest.raises(RuntimeError, match="already"):
                runner.start_run("Groww", "2026-08-10", "2026-09-07")

    def test_pipeline_parks_run_for_approval(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")

        class FakeGraph:
            def stream(self, state, stream_mode=None):
                yield {"cluster": {"clusters": [{"label": "Crashes",
                                                 "review_ids": ["r1", "r2"],
                                                 "centroid_quotes": ["boom"]}]}}
                yield {"generate_report": {"report_markdown": "## Summary\ntext",
                                           "cleaned_reviews": [{}] * 30}}
                yield {"generate_pdf": {"pdf_path": "data/reports/x.pdf",
                                        "insights": {}}}

        with patch("src.graph.build_graph", return_value=FakeGraph()):
            runner._run_pipeline(run_id, "Groww", "2026-08-10", "2026-09-07")

        run = store.get_run(run_id)
        assert run["status"] == store.AWAITING_APPROVAL
        assert run["report_markdown"] == "## Summary\ntext"
        assert run["theme_count"] == 1
        assert run["pdf_path"] == "data/reports/x.pdf"

    def test_pipeline_uses_a_graph_without_delivery(self, db):
        """The generate phase must be structurally unable to deliver."""
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")

        class FakeGraph:
            def stream(self, state, stream_mode=None):
                return iter([])

        with patch("src.graph.build_graph", return_value=FakeGraph()) as mock_build:
            runner._run_pipeline(run_id, "Groww", "2026-08-10", "2026-09-07")

        assert mock_build.call_args.kwargs["include_delivery"] is False

    def test_pipeline_failure_is_recorded(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")

        with patch("src.graph.build_graph", side_effect=RuntimeError("boom")):
            runner._run_pipeline(run_id, "Groww", "2026-08-10", "2026-09-07")

        run = store.get_run(run_id)
        assert run["status"] == store.FAILED
        assert "boom" in run["error"]

    def test_approve_requires_awaiting_status(self, db):
        run_id = store.create_run("Groww", "2026-08-10", "2026-09-07")
        with pytest.raises(RuntimeError, match="not awaiting approval"):
            runner.approve(run_id)

    def test_approve_unknown_run(self, db):
        with pytest.raises(LookupError):
            runner.approve("nope")

    def test_delivery_passes_stored_content(self, db):
        run_id = _finished_run()
        captured = {}

        def fake_deliver(state):
            captured.update(state)
            return {"pdf_url": "https://drive.test/x", "email_sent": True,
                    "audit_record": {}}

        with patch("src.nodes.deliver.deliver", fake_deliver):
            runner._run_delivery(run_id)

        assert len(captured["cleaned_reviews"]) == 4821
        assert len(captured["clusters"]) == 6
        assert captured["report_markdown"] == "## Summary\nUsers report crashes."

        run = store.get_run(run_id)
        assert run["status"] == store.DELIVERED
        assert run["pdf_url"] == "https://drive.test/x"
        assert run["email_sent"] is True

    def test_delivery_failure_is_recorded(self, db):
        run_id = _finished_run()
        with patch("src.nodes.deliver.deliver", side_effect=RuntimeError("502")):
            runner._run_delivery(run_id)

        run = store.get_run(run_id)
        assert run["status"] == store.FAILED
        assert "502" in run["error"]

    def test_reject_marks_run_and_blocks_later_approval(self, db):
        run_id = _finished_run()
        runner.reject(run_id)

        assert store.get_run(run_id)["status"] == store.REJECTED
        with pytest.raises(RuntimeError):
            runner.approve(run_id)


# ──────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────


class TestApi:
    def test_health(self, client):
        body = client.get("/api/health").json()
        assert body["status"] == "ok" and body["busy"] is False

    def test_index_serves_the_ui(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "Pulse" in res.text

    def test_trigger_defaults_to_eight_week_window(self, client):
        with patch("src.web.runner._spawn"):
            res = client.post("/api/runs", json={})

        assert res.status_code == 202
        body = res.json()
        assert body["product"] == "Groww"
        assert body["week_start"] < body["week_end"]

    def test_trigger_accepts_explicit_window(self, client):
        with patch("src.web.runner._spawn"):
            res = client.post("/api/runs", json={
                "product": "Other", "week_start": "2026-07-01",
                "week_end": "2026-08-01"})

        body = res.json()
        assert body["product"] == "Other"
        assert body["week_start"] == "2026-07-01"

    def test_trigger_rejects_bad_dates(self, client):
        assert client.post("/api/runs", json={"week_end": "not-a-date"}).status_code == 400

    def test_trigger_rejects_inverted_window(self, client):
        res = client.post("/api/runs", json={"week_start": "2026-09-01",
                                             "week_end": "2026-08-01"})
        assert res.status_code == 400

    def test_second_trigger_conflicts(self, client):
        with patch("src.web.runner._spawn"):
            client.post("/api/runs", json={})
            assert client.post("/api/runs", json={}).status_code == 409

    def test_history_omits_bulky_fields(self, client):
        _finished_run()
        runs = client.get("/api/runs").json()["runs"]
        assert runs and "report_markdown" not in runs[0]
        assert runs[0]["review_count"] == 4821

    def test_detail_includes_the_report(self, client):
        run_id = _finished_run()
        body = client.get(f"/api/runs/{run_id}").json()
        assert body["report_markdown"].startswith("## Summary")
        assert body["themes"][0]["label"] == "Crashes"

    def test_detail_404(self, client):
        assert client.get("/api/runs/nope").status_code == 404

    def test_approve_moves_to_delivering(self, client):
        run_id = _finished_run()
        with patch("src.web.runner._spawn"):
            res = client.post(f"/api/runs/{run_id}/approve")

        assert res.status_code == 200
        assert store.get_run(run_id)["approved_at"]

    def test_approve_wrong_state_conflicts(self, client):
        run_id = _finished_run(status=store.DELIVERED)
        assert client.post(f"/api/runs/{run_id}/approve").status_code == 409

    def test_approve_unknown_run_404(self, client):
        assert client.post("/api/runs/nope/approve").status_code == 404

    def test_reject(self, client):
        run_id = _finished_run()
        assert client.post(f"/api/runs/{run_id}/reject").status_code == 200
        assert store.get_run(run_id)["status"] == store.REJECTED

    def test_pdf_download_404_when_missing(self, client):
        run_id = _finished_run(pdf_path="/does/not/exist.pdf")
        assert client.get(f"/api/runs/{run_id}/pdf").status_code == 404

    def test_pdf_download_serves_the_file(self, client, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 test")
        run_id = _finished_run(pdf_path=str(pdf))

        res = client.get(f"/api/runs/{run_id}/pdf")
        assert res.status_code == 200
        assert res.content.startswith(b"%PDF-")

    def test_triggering_never_delivers(self, client):
        """The trigger path must not touch the delivery node at all."""
        with patch("src.nodes.deliver.deliver") as mock_deliver:
            with patch("src.web.runner._spawn"):
                client.post("/api/runs", json={})
        mock_deliver.assert_not_called()
