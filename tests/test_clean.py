"""
Tests for Sub-Phase 2a — clean node.

Covers: Unicode normalisation, HTML/URL stripping, language filter,
minimum length gate, original_text preservation, langdetect exception handling.
"""

from unittest.mock import patch

import pytest

from src.nodes.clean import (
    _is_english,
    _normalise_unicode,
    _strip_html_urls,
    clean,
)


# ── Helper ───────────────────────────────────────────────────

def _make_raw(review_id="r1", text="This is a great trading application for beginners", rating=5):
    """Create a raw review dict in the Phase 1 output format."""
    return {
        "review_id": review_id,
        "author_name": "TestUser",
        "text": text,
        "star_rating": rating,
        "date": "2026-08-15",
        "thumbs_up_count": 5,
        "app_version": "5.6.1",
    }


# ── Unicode normalisation ────────────────────────────────────

class TestUnicodeNormalisation:

    def test_nfkc_normalisation(self):
        # Fullwidth characters → ASCII
        assert _normalise_unicode("\uff21\uff22\uff23") == "ABC"

    def test_strips_non_printable(self):
        result = _normalise_unicode("hello\x00world\x01test")
        assert "\x00" not in result
        assert "\x01" not in result
        assert "hello" in result

    def test_strips_emoji(self):
        result = _normalise_unicode("Great app 🚀🔥👍")
        assert "🚀" not in result
        assert "Great app" in result

    def test_accented_characters(self):
        # Accented chars may be stripped by the non-printable filter
        # but NFKC handles decomposed forms
        result = _normalise_unicode("café")
        # 'é' (U+00E9) is outside \x20-\x7E, so gets stripped
        assert "caf" in result


# ── HTML / URL stripping ─────────────────────────────────────

class TestStripHtmlUrls:

    def test_removes_html_tags(self):
        result = _strip_html_urls("<b>bold</b> and <i>italic</i>")
        assert "<b>" not in result
        assert "<i>" not in result
        assert "bold" in result
        assert "italic" in result

    def test_removes_http_urls(self):
        result = _strip_html_urls("Visit https://example.com for details")
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_removes_www_urls(self):
        result = _strip_html_urls("Check www.groww.in for more")
        assert "www.groww.in" not in result
        assert "Check" in result

    def test_plain_text_unchanged(self):
        text = "This is a normal review without any links"
        assert _strip_html_urls(text) == text


# ── Language filter ──────────────────────────────────────────

class TestLanguageFilter:

    def test_english_detected(self):
        assert _is_english("This app is amazing and works very well") is True

    def test_hindi_rejected(self):
        assert _is_english("यह ऐप बहुत अच्छा है") is False

    def test_spanish_rejected(self):
        assert _is_english("Esta aplicación es muy buena") is False

    def test_empty_string_rejected(self):
        assert _is_english("") is False

    def test_whitespace_only_rejected(self):
        assert _is_english("   ") is False

    def test_langdetect_exception_handled(self):
        """langdetect raises on ambiguous input; should return False."""
        # Very short text can cause LangDetectException
        with patch("src.nodes.clean.detect", side_effect=Exception("detection error")):
            assert _is_english("x") is False


# ── clean() node tests ───────────────────────────────────────

class TestCleanNode:

    def test_basic_cleaning(self):
        state = {"raw_reviews": [_make_raw()]}
        result = clean(state)

        assert len(result["cleaned_reviews"]) == 1
        review = result["cleaned_reviews"][0]
        assert review["review_id"] == "r1"
        assert review["rating"] == 5
        assert review["date"] == "2026-08-15"

    def test_original_text_preserved(self):
        """original_text retains casing; cleaned_text is lowered."""
        state = {"raw_reviews": [_make_raw(text="Great App For Trading!")]}
        result = clean(state)

        review = result["cleaned_reviews"][0]
        assert review["original_text"] == "Great App For Trading!"
        assert review["cleaned_text"] == "great app for trading!"

    def test_min_length_gate_discards_short(self):
        """Reviews shorter than 10 chars are discarded."""
        state = {"raw_reviews": [
            _make_raw(review_id="short", text="good"),      # 4 chars → discard
            _make_raw(review_id="long", text="This application is really wonderful"),  # passes
        ]}
        result = clean(state)

        assert len(result["cleaned_reviews"]) == 1
        assert result["cleaned_reviews"][0]["review_id"] == "long"

    def test_non_english_discarded(self):
        """Non-English reviews are filtered out."""
        state = {"raw_reviews": [
            _make_raw(review_id="en", text="This is a wonderful application for stock trading"),
            _make_raw(review_id="hi", text="यह ऐप बहुत अच्छा है और बहुत अच्छा काम करता है"),
        ]}
        result = clean(state)

        ids = [r["review_id"] for r in result["cleaned_reviews"]]
        assert "en" in ids
        assert "hi" not in ids

    def test_html_stripped(self):
        """HTML tags are removed from review text."""
        state = {"raw_reviews": [
            _make_raw(text="<b>Great</b> application for trading and investing")
        ]}
        result = clean(state)

        review = result["cleaned_reviews"][0]
        assert "<b>" not in review["cleaned_text"]
        assert "great" in review["cleaned_text"]

    def test_url_stripped(self):
        """URLs are removed from review text."""
        state = {"raw_reviews": [
            _make_raw(text="Check https://groww.in for more information about this app")
        ]}
        result = clean(state)

        review = result["cleaned_reviews"][0]
        assert "https://groww.in" not in review["cleaned_text"]

    def test_empty_text_discarded(self):
        """Reviews with empty text are discarded."""
        state = {"raw_reviews": [
            _make_raw(review_id="empty", text=""),
            _make_raw(review_id="whitespace", text="   "),
            _make_raw(review_id="valid", text="This is a valid review for testing"),
        ]}
        result = clean(state)

        ids = [r["review_id"] for r in result["cleaned_reviews"]]
        assert "empty" not in ids
        assert "whitespace" not in ids
        assert "valid" in ids

    def test_empty_input(self):
        """Empty raw_reviews returns empty cleaned_reviews."""
        result = clean({"raw_reviews": []})
        assert result["cleaned_reviews"] == []

    def test_field_mapping(self):
        """Verify all output fields are correctly mapped."""
        state = {"raw_reviews": [_make_raw()]}
        result = clean(state)
        review = result["cleaned_reviews"][0]

        assert set(review.keys()) == {
            "review_id", "original_text", "cleaned_text",
            "rating", "date", "author_name",
            "thumbs_up_count", "app_version",
        }

    def test_text_becomes_empty_after_stripping(self):
        """If text becomes empty after HTML/URL stripping, discard."""
        state = {"raw_reviews": [
            _make_raw(text="<a>https://example.com</a>"),
        ]}
        result = clean(state)
        assert len(result["cleaned_reviews"]) == 0
