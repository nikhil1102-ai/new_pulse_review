"""
Unit tests for Phase 7 — Delivery & Audit.

Tests cover:
  - Local report saving
  - Audit log writing and loading
  - Idempotency (skipping duplicate delivery)
  - Graceful MCP fallback when server is unavailable
  - Full deliver node integration
"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock


def _make_review(review_id, text, rating=4):
    """Helper to create a review dict."""
    return {
        "review_id": review_id,
        "original_text": text,
        "cleaned_text": text.lower(),
        "rating": rating,
        "date": "2026-08-15",
        "author_name": "TestUser",
        "thumbs_up_count": 0,
        "app_version": "5.0.0",
    }


SAMPLE_REPORT = """\
## Summary of Top Themes

Groww users report crashes and slow withdrawals.

## Supporting User Quotes

> The app crashes every morning when the market opens
> My withdrawal has been stuck for 3 days
> Customer support is not responsive

## Key Observation

App stability at market open is the most critical issue.

## Action Ideas

1. Fix crashes
2. Speed up withdrawals
3. Improve support SLA
"""


class TestReportHash:
    """Tests for ``_compute_report_hash``."""

    def test_deterministic_hash(self):
        from src.nodes.deliver import _compute_report_hash

        h1 = _compute_report_hash("Hello World")
        h2 = _compute_report_hash("Hello World")

        assert h1 == h2
        assert len(h1) == 16

    def test_different_content_different_hash(self):
        from src.nodes.deliver import _compute_report_hash

        h1 = _compute_report_hash("Report A")
        h2 = _compute_report_hash("Report B")

        assert h1 != h2


class TestAuditLog:
    """Tests for audit log save/load."""

    def test_save_and_load(self, tmp_path):
        from src.nodes.deliver import _save_audit_log, _load_audit_log

        audit = {
            "product": "Groww",
            "week_start": "2026-08-01",
            "email_sent": True,
        }

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            _save_audit_log(audit, "Groww", "2026-08-01")
            loaded = _load_audit_log("Groww", "2026-08-01")

        assert loaded is not None
        assert loaded["product"] == "Groww"
        assert loaded["email_sent"] is True

    def test_load_nonexistent_returns_none(self, tmp_path):
        from src.nodes.deliver import _load_audit_log

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            result = _load_audit_log("Groww", "2099-01-01")

        assert result is None


class TestLocalReportSave:
    """Tests for ``_save_report_locally``."""

    def test_saves_to_disk(self, tmp_path):
        from src.nodes.deliver import _save_report_locally

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            path = _save_report_locally(SAMPLE_REPORT, "Groww", "2026-08-01")

        assert os.path.exists(path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Summary of Top Themes" in content


class TestGracefulFallbacks:
    """Tests for MCP graceful fallbacks."""

    def test_google_docs_skipped_without_mcp_url(self):
        from src.nodes.deliver import _create_google_doc_via_mcp

        with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
            result = _create_google_doc_via_mcp(
                SAMPLE_REPORT, "Groww", "2026-08-01"
            )

        assert result is None

    def test_gmail_skipped_without_recipients(self):
        from src.nodes.deliver import _send_email_via_mcp

        with patch("src.nodes.deliver.REPORT_RECIPIENTS", []):
            result = _send_email_via_mcp(
                SAMPLE_REPORT, "Groww", "2026-08-01", None
            )

        assert result is False

    def test_gmail_skipped_without_mcp_url(self):
        from src.nodes.deliver import _send_email_via_mcp

        with patch("src.nodes.deliver.REPORT_RECIPIENTS", ["test@example.com"]):
            with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
                result = _send_email_via_mcp(
                    SAMPLE_REPORT, "Groww", "2026-08-01", None
                )

        assert result is False


class TestDeliverNode:
    """Tests for the full ``deliver`` node."""

    def test_delivers_report_locally(self, tmp_path):
        from src.nodes.deliver import deliver

        state = {
            "report_markdown": SAMPLE_REPORT,
            "fee_explainer": "",
            "fee_pain_point": "",
            "cleaned_reviews": [_make_review("r1", "Review")],
            "clusters": [{"label": "Theme", "review_ids": ["r1"]}],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "validation_passed": True,
            "validation_errors": [],
        }

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
                result = deliver(state)

        assert "audit_record" in result
        assert result["audit_record"]["product"] == "Groww"
        assert result["audit_record"]["review_count"] == 1
        assert result["audit_record"]["theme_count"] == 1
        assert result["email_sent"] is False
        assert result["doc_url"] is None

        # Verify files were created
        files = os.listdir(tmp_path)
        report_files = [f for f in files if f.startswith("report_")]
        audit_files = [f for f in files if f.startswith("audit_")]
        assert len(report_files) == 1
        assert len(audit_files) == 1

    def test_idempotent_delivery_skips(self, tmp_path):
        """Re-running delivery for the same week should skip."""
        from src.nodes.deliver import deliver

        # Create a pre-existing audit log
        audit = {
            "product": "Groww",
            "week_start": "2026-08-01",
            "doc_url": "https://docs.google.com/existing",
            "email_sent": True,
        }
        audit_path = tmp_path / "audit_Groww_2026-08-01.json"
        with open(audit_path, "w") as f:
            json.dump(audit, f)

        state = {
            "report_markdown": SAMPLE_REPORT,
            "fee_explainer": "",
            "fee_pain_point": "",
            "cleaned_reviews": [],
            "clusters": [],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "validation_passed": True,
            "validation_errors": [],
        }

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            result = deliver(state)

        assert result["doc_url"] == "https://docs.google.com/existing"
        assert result["email_sent"] is True

    def test_audit_record_contains_required_fields(self, tmp_path):
        from src.nodes.deliver import deliver

        state = {
            "report_markdown": SAMPLE_REPORT,
            "fee_explainer": "",
            "fee_pain_point": "",
            "cleaned_reviews": [_make_review("r1", "Review")],
            "clusters": [{"label": "Theme", "review_ids": ["r1"]}],
            "product": "Groww",
            "week_start": "2026-08-01",
            "week_end": "2026-08-07",
            "validation_passed": True,
            "validation_errors": [],
        }

        with patch("src.nodes.deliver.REPORTS_DIR", str(tmp_path)):
            with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
                result = deliver(state)

        audit = result["audit_record"]
        required_keys = [
            "product", "week_start", "week_end", "review_count",
            "theme_count", "report_hash", "report_path", "doc_url",
            "email_sent", "validation_passed", "timestamp",
            "fee_pain_point", "has_fee_explainer",
        ]
        for key in required_keys:
            assert key in audit, f"Missing audit key: {key}"
