"""
Tests for Sub-Phase 2c — pii_scrub node.

Covers: phone, email, ID, and name redaction; no-PII pass-through;
multiple PII in one review; both fields scrubbed; file persistence.
"""

import json
import os
from unittest.mock import patch

import pytest

from src.nodes.pii_scrub import _scrub_text, pii_scrub


# ── Helper ───────────────────────────────────────────────────

def _make_review(review_id="r1", cleaned_text="great app", original_text=None):
    return {
        "review_id": review_id,
        "original_text": original_text or cleaned_text.title(),
        "cleaned_text": cleaned_text,
        "rating": 5,
        "date": "2026-08-15",
        "author_name": "TestUser",
        "thumbs_up_count": 5,
        "app_version": "5.6.1",
    }


# ── _scrub_text unit tests ──────────────────────────────────

class TestScrubText:

    def test_phone_redaction(self):
        text, counts = _scrub_text("call me at +91 98765 43210 please")
        assert "[PHONE]" in text
        assert "+91 98765 43210" not in text
        assert counts.get("phone", 0) >= 1

    def test_phone_with_dashes(self):
        text, counts = _scrub_text("my number is 98765-43210 thanks")
        assert "[PHONE]" in text
        assert "98765-43210" not in text

    def test_email_redaction(self):
        text, counts = _scrub_text("contact user@example.com for help")
        assert "[EMAIL]" in text
        assert "user@example.com" not in text
        assert counts.get("email", 0) >= 1

    def test_id_redaction(self):
        text, counts = _scrub_text("my order ORD123456789 is pending")
        assert "[ID]" in text
        assert "ORD123456789" not in text
        assert counts.get("id", 0) >= 1

    def test_upi_id_redaction(self):
        """UPI-style IDs like UPI123456 should be caught."""
        text, _ = _scrub_text("transaction UPI123456 failed")
        assert "[ID]" in text

    def test_name_redaction_my_name_is(self):
        text, counts = _scrub_text("my name is John Smith and I need help")
        assert "[NAME]" in text
        assert "John Smith" not in text
        assert counts.get("name", 0) >= 1

    def test_name_redaction_i_am(self):
        text, counts = _scrub_text("I am Rahul and I have a problem")
        assert "[NAME]" in text
        assert "Rahul" not in text

    def test_no_pii(self):
        """Text without PII passes through unchanged."""
        original = "this is a normal review about the app"
        text, counts = _scrub_text(original)
        assert text == original
        assert counts == {}

    def test_multiple_pii(self):
        """Multiple PII types in one text are all redacted."""
        text, counts = _scrub_text(
            "call +91 98765 43210 or email user@example.com order ORD123456789"
        )
        assert "[PHONE]" in text
        assert "[EMAIL]" in text
        assert "[ID]" in text


# ── pii_scrub node tests ────────────────────────────────────

class TestPiiScrubNode:

    def test_scrubs_both_fields(self):
        """PII is redacted from both cleaned_text and original_text."""
        state = {
            "cleaned_reviews": [
                _make_review(
                    cleaned_text="call +91 98765 43210 for support",
                    original_text="Call +91 98765 43210 For Support",
                )
            ],
            "week_start": "2026-07-16",
        }
        with patch("src.nodes.pii_scrub.CLEAN_DATA_DIR", os.path.join("data", "clean")):
            result = pii_scrub(state)

        review = result["cleaned_reviews"][0]
        assert "+91 98765 43210" not in review["cleaned_text"]
        assert "+91 98765 43210" not in review["original_text"]
        assert "[PHONE]" in review["cleaned_text"]
        assert "[PHONE]" in review["original_text"]

    def test_no_pii_passthrough(self):
        """Reviews without PII are unchanged."""
        state = {
            "cleaned_reviews": [
                _make_review(
                    cleaned_text="this app is really great for trading",
                    original_text="This App Is Really Great For Trading",
                )
            ],
            "week_start": "2026-07-16",
        }
        with patch("src.nodes.pii_scrub.CLEAN_DATA_DIR", os.path.join("data", "clean")):
            result = pii_scrub(state)

        review = result["cleaned_reviews"][0]
        assert review["cleaned_text"] == "this app is really great for trading"
        assert review["original_text"] == "This App Is Really Great For Trading"

    def test_persists_to_disk(self, tmp_path):
        """Final corpus is written to data/clean/."""
        state = {
            "cleaned_reviews": [
                _make_review(cleaned_text="a normal review without any personal info")
            ],
            "week_start": "2026-07-16",
        }
        with patch("src.nodes.pii_scrub.CLEAN_DATA_DIR", str(tmp_path)):
            result = pii_scrub(state)

        output_file = tmp_path / "clean_reviews_2026-07-16.json"
        assert output_file.exists()

        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["review_id"] == "r1"

    def test_does_not_mutate_input(self):
        """The original state dict is not mutated."""
        original_review = _make_review(
            cleaned_text="email me at user@example.com please",
            original_text="Email Me At user@example.com Please",
        )
        original_text_before = original_review["cleaned_text"]

        state = {
            "cleaned_reviews": [original_review],
            "week_start": "2026-07-16",
        }
        with patch("src.nodes.pii_scrub.CLEAN_DATA_DIR", os.path.join("data", "clean")):
            pii_scrub(state)

        # Original dict should not be mutated
        assert original_review["cleaned_text"] == original_text_before

    def test_empty_input(self, tmp_path):
        """Empty input produces empty output and an empty JSON file."""
        state = {"cleaned_reviews": [], "week_start": "2026-07-16"}
        with patch("src.nodes.pii_scrub.CLEAN_DATA_DIR", str(tmp_path)):
            result = pii_scrub(state)

        assert result["cleaned_reviews"] == []
        output_file = tmp_path / "clean_reviews_2026-07-16.json"
        assert output_file.exists()
        assert json.loads(output_file.read_text()) == []
