"""
Unit tests for the two-stage (generate / deliver) split.

Covers the graph builder's optional delivery node and the handoff file that
carries a generated run across the approval gate into the delivery stage.
"""

import json
import logging
import os
from unittest.mock import patch

import pytest

import src.main as main_module
from src.graph import build_graph


LOGGER = logging.getLogger("test")


# ──────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────


class TestBuildGraph:
    def test_full_graph_includes_delivery(self):
        nodes = build_graph(include_delivery=True).get_graph().nodes
        assert "deliver" in nodes
        assert "generate_pdf" in nodes

    def test_generate_only_graph_excludes_delivery(self):
        """The generate stage must not be able to reach an external call."""
        nodes = build_graph(include_delivery=False).get_graph().nodes
        assert "deliver" not in nodes
        assert "generate_pdf" in nodes

    def test_both_graphs_share_every_other_node(self):
        full = set(build_graph(True).get_graph().nodes)
        gen = set(build_graph(False).get_graph().nodes)
        assert full - gen == {"deliver"}


# ──────────────────────────────────────────────────────────────
# Handoff file
# ──────────────────────────────────────────────────────────────


@pytest.fixture
def final_state(tmp_path) -> dict:
    pdf = tmp_path / "groww_weekly_pulse_2026-08-10.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return {
        "product": "Groww",
        "week_start": "2026-08-10",
        "week_end": "2026-09-07",
        "report_markdown": "## Summary\nUsers report crashes.",
        "fee_explainer": "- Brokerage is zero.",
        "fee_pain_point": "Unexplained deductions",
        "pdf_path": str(pdf),
        "validation_passed": True,
        "validation_errors": [],
        "cleaned_reviews": [{"review_id": f"r{i}"} for i in range(4821)],
        "clusters": [{"label": f"t{i}"} for i in range(6)],
    }


class TestSavePending:
    def test_writes_expected_fields(self, final_state, tmp_path):
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            path = main_module._save_pending(final_state, LOGGER)

        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["product"] == "Groww"
        assert payload["review_count"] == 4821
        assert payload["theme_count"] == 6
        assert payload["report_markdown"] == final_state["report_markdown"]
        assert payload["generated_at"]

    def test_does_not_serialise_the_corpus(self, final_state, tmp_path):
        """The handoff travels as a CI artifact - it must stay small."""
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            path = main_module._save_pending(final_state, LOGGER)

        payload = json.loads(open(path, encoding="utf-8").read())
        assert "cleaned_reviews" not in payload
        assert "clusters" not in payload
        assert os.path.getsize(path) < 20_000

    def test_handles_missing_optional_fields(self, tmp_path):
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            path = main_module._save_pending({"product": "X"}, LOGGER)

        payload = json.loads(open(path, encoding="utf-8").read())
        assert payload["review_count"] == 0
        assert payload["pdf_path"] is None


class TestRunDeliver:
    def test_rebuilds_counts_and_passes_content(self, final_state, tmp_path):
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            main_module._save_pending(final_state, LOGGER)

            captured = {}

            def fake_deliver(state):
                captured.update(state)
                return {"pdf_url": "https://drive.test/x", "email_sent": True,
                        "audit_record": {}}

            with patch("src.nodes.deliver.deliver", fake_deliver):
                main_module._run_deliver(LOGGER)

        assert len(captured["cleaned_reviews"]) == 4821
        assert len(captured["clusters"]) == 6
        assert captured["report_markdown"] == final_state["report_markdown"]
        assert captured["fee_explainer"] == final_state["fee_explainer"]
        assert captured["pdf_path"] == final_state["pdf_path"]

    def test_exits_when_no_pending_run(self, tmp_path):
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            with pytest.raises(SystemExit) as exc:
                main_module._run_deliver(LOGGER)
        assert exc.value.code == 1

    def test_exits_when_delivery_raises(self, final_state, tmp_path):
        with patch("src.main.REPORTS_DIR", str(tmp_path)):
            main_module._save_pending(final_state, LOGGER)
            with patch("src.nodes.deliver.deliver", side_effect=RuntimeError("boom")):
                with pytest.raises(SystemExit) as exc:
                    main_module._run_deliver(LOGGER)
        assert exc.value.code == 1


# ──────────────────────────────────────────────────────────────
# CLI wiring
# ──────────────────────────────────────────────────────────────


class TestStageArgument:
    def test_defaults_to_all(self):
        with patch("sys.argv", ["prog"]):
            assert main_module._parse_args().stage == "all"

    @pytest.mark.parametrize("stage", ["all", "generate", "deliver"])
    def test_accepts_each_stage(self, stage):
        with patch("sys.argv", ["prog", "--stage", stage]):
            assert main_module._parse_args().stage == stage

    def test_rejects_unknown_stage(self):
        with patch("sys.argv", ["prog", "--stage", "nonsense"]):
            with pytest.raises(SystemExit):
                main_module._parse_args()

    def test_deliver_stage_skips_the_graph(self, tmp_path):
        """--stage deliver must not import or build the pipeline graph."""
        with patch("sys.argv", ["prog", "--stage", "deliver"]):
            with patch("src.main._run_deliver") as mock_deliver:
                with patch("src.graph.build_graph") as mock_build:
                    main_module.main()

        mock_deliver.assert_called_once()
        mock_build.assert_not_called()
