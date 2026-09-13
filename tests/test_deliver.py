"""
Unit tests for Phase 7 — Delivery & Audit.

Tests cover:
  - Local report saving
  - Audit log writing and loading
  - Idempotency (skipping duplicate delivery)
  - Graceful MCP fallback when server is unavailable
  - Full deliver node integration
"""

import base64
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
    """Delivery must degrade quietly when the REST server is unconfigured."""

    def test_pdf_upload_skipped_without_server_url(self, tmp_path):
        from src.nodes.deliver import _upload_pdf_via_rest

        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
            result = _upload_pdf_via_rest(str(pdf))

        assert result is None

    def test_pdf_upload_skipped_without_pdf(self):
        """A missing PDF must not raise — email delivery still proceeds."""
        from src.nodes.deliver import _upload_pdf_via_rest

        with patch("src.nodes.deliver.MCP_SERVER_URL", "https://server.test"):
            assert _upload_pdf_via_rest(None) is None
            assert _upload_pdf_via_rest("/does/not/exist.pdf") is None

    def test_pdf_upload_returns_url_from_server(self, tmp_path):
        from src.nodes.deliver import _upload_pdf_via_rest

        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("src.nodes.deliver.MCP_SERVER_URL", "https://server.test"):
            with patch(
                "src.nodes.deliver.upload_to_drive",
                return_value={"success": True,
                              "data": {"file_url": "https://drive.test/abc"}},
            ) as mock_upload:
                result = _upload_pdf_via_rest(str(pdf))

        assert result == "https://drive.test/abc"
        assert mock_upload.call_args.kwargs["filename"] == "report.pdf"
        assert mock_upload.call_args.kwargs["content"] == b"%PDF-1.4 fake"

    def test_gmail_skipped_without_recipients(self):
        from src.nodes.deliver import _send_email_via_rest

        with patch("src.nodes.deliver.REPORT_RECIPIENTS", []):
            result = _send_email_via_rest(
                SAMPLE_REPORT, "Groww", "2026-08-01", None
            )

        assert result is False

    def test_gmail_skipped_without_mcp_url(self):
        from src.nodes.deliver import _send_email_via_rest

        with patch("src.nodes.deliver.REPORT_RECIPIENTS", ["test@example.com"]):
            with patch("src.nodes.deliver.MCP_SERVER_URL", ""):
                result = _send_email_via_rest(
                    SAMPLE_REPORT, "Groww", "2026-08-01", None
                )

        assert result is False

    def test_email_subject_and_pdf_link(self):
        """Subject must match the spec and the body must carry the PDF link."""
        from src.nodes.deliver import _send_email_via_rest

        with patch("src.nodes.deliver.REPORT_RECIPIENTS", ["test@example.com"]):
            with patch("src.nodes.deliver.MCP_SERVER_URL", "https://server.test"):
                with patch(
                    "src.nodes.deliver.create_email_draft",
                    return_value={"success": True, "data": {}},
                ) as mock_draft:
                    sent = _send_email_via_rest(
                        SAMPLE_REPORT, "Groww", "2026-08-01",
                        "https://drive.test/abc",
                    )

        assert sent is True
        kwargs = mock_draft.call_args.kwargs
        assert kwargs["subject"].startswith(
            "Weekly Product Pulse + Customer Clarification"
        )
        assert "https://drive.test/abc" in kwargs["body"]
        assert "https://drive.test/abc" in kwargs["body_html"]


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
        assert result["pdf_url"] is None

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
            "pdf_url": "https://drive.google.com/existing",
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

        assert result["pdf_url"] == "https://drive.google.com/existing"
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
            "theme_count", "report_hash", "report_path", "pdf_url",
            "pdf_path",
            "email_sent", "validation_passed", "timestamp",
            "fee_pain_point", "has_fee_explainer",
        ]
        for key in required_keys:
            assert key in audit, f"Missing audit key: {key}"


class TestUploadToDriveClient:
    """Tests for the REST client helper that ships the PDF."""

    def test_encodes_payload_and_posts_to_correct_endpoint(self):
        from src.mcp_client import upload_to_drive

        with patch("src.mcp_client._post", return_value={"success": True}) as mock_post:
            upload_to_drive("report.pdf", b"%PDF-1.4 body")

        endpoint, payload = mock_post.call_args.args
        assert endpoint == "/upload_to_drive"
        assert payload["filename"] == "report.pdf"
        assert payload["mime_type"] == "application/pdf"
        assert base64.b64decode(payload["content_b64"]) == b"%PDF-1.4 body"

    def test_raises_without_server_url(self):
        from src.mcp_client import upload_to_drive

        with patch("src.mcp_client.MCP_SERVER_URL", ""):
            with pytest.raises(RuntimeError, match="MCP_SERVER_URL"):
                upload_to_drive("report.pdf", b"data")
