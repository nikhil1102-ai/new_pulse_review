"""
Phase 7 — Deliver report via REST server (Google Docs & Gmail) and write audit log.

Channels:
1. **Google Docs** — append the report to a Google Doc via ``POST /append_to_doc``
   on the Railway-deployed FastAPI server
2. **Gmail** — create an email draft via ``POST /create_email_draft`` on the same server
3. **Audit log** — persist a JSON record of the run to ``data/reports/``

The FastAPI server has OAuth credentials and token configuration embedded, so
delivery requires only the ``MCP_SERVER_URL`` environment variable (no /sse suffix).

All delivery channels are skipped gracefully if the server URL is not
configured, falling back to local-only report persistence + audit log.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from src.config import (
    MCP_SERVER_URL,
    REPORT_RECIPIENTS,
    REPORTS_DIR,
)
from src.mcp_client import append_to_doc, create_email_draft
from src.state import PipelineState

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────


def _compute_report_hash(report_markdown: str) -> str:
    """SHA-256 hash of the report content (first 16 hex chars)."""
    return hashlib.sha256(report_markdown.encode("utf-8")).hexdigest()[:16]


def _load_audit_log(product: str, week_start: str) -> dict | None:
    """Load an existing audit log for this product/week, if any."""
    audit_path = os.path.join(REPORTS_DIR, f"audit_{product}_{week_start}.json")
    if os.path.exists(audit_path):
        with open(audit_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_audit_log(audit_record: dict, product: str, week_start: str) -> str:
    """Persist the audit record to disk."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    audit_path = os.path.join(REPORTS_DIR, f"audit_{product}_{week_start}.json")

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_record, f, indent=2, default=str)

    logger.info("Audit log written to %s", audit_path)
    return audit_path


def _save_report_locally(
    report_markdown: str, product: str, week_start: str
) -> str:
    """Save the report Markdown to the reports directory."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(
        REPORTS_DIR, f"report_{product}_{week_start}.md"
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)

    logger.info("Report saved locally to %s", report_path)
    return report_path


# ── Email HTML builder ──────────────────────────────────────


def _build_html_email(
    report_markdown: str,
    product: str,
    week_start: str,
    doc_url: str | None,
) -> str:
    """Convert the Markdown report to a styled HTML email body.

    Handles:
    - ``## Heading`` → bold ``<h3>`` section headers
    - ``> quote``    → styled blockquote with left border
    - ``N. item``    → ``<ol>`` numbered list
    - ``---``        → ``<hr>`` divider
    - Plain text     → ``<p>`` paragraph

    A clickable Google Doc button is injected at the top when *doc_url*
    is provided.
    """
    import re

    html_parts: list[str] = []
    in_ol = False

    def _close_ol() -> None:
        nonlocal in_ol
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    for line in report_markdown.split("\n"):
        # ── H2 section header ─────────────────────────────────
        m = re.match(r"^##\s+(.+)", line)
        if m:
            _close_ol()
            html_parts.append(
                f'<h3 style="color:#1a1a2e;margin:28px 0 6px;'
                f'padding-bottom:6px;border-bottom:2px solid #e8e8e8;">'
                f"<strong>{m.group(1)}</strong></h3>"
            )
            continue

        # ── Blockquote (user quotes) ──────────────────────────
        m = re.match(r"^>\s*(.*)", line)
        if m:
            _close_ol()
            quote = m.group(1).strip('"')
            html_parts.append(
                f'<blockquote style="border-left:4px solid #2196F3;'
                f"margin:10px 0;padding:10px 16px;background:#f0f7ff;"
                f'border-radius:0 6px 6px 0;font-style:italic;color:#555;">'
                f"&ldquo;{quote}&rdquo;</blockquote>"
            )
            continue

        # ── Numbered list item ────────────────────────────────
        m = re.match(r"^\d+\.\s+(.+)", line)
        if m:
            if not in_ol:
                html_parts.append(
                    '<ol style="padding-left:22px;margin:8px 0;">'
                )
                in_ol = True
            html_parts.append(
                f'<li style="margin-bottom:8px;line-height:1.6;">'
                f"{m.group(1)}</li>"
            )
            continue

        # ── Horizontal rule ───────────────────────────────────
        if line.strip() == "---":
            _close_ol()
            html_parts.append(
                '<hr style="border:none;border-top:1px solid #e8e8e8;margin:20px 0;">'
            )
            continue

        # ── Empty line ────────────────────────────────────────
        if line.strip() == "":
            _close_ol()
            continue

        # ── Normal paragraph ──────────────────────────────────
        _close_ol()
        html_parts.append(
            f'<p style="margin:6px 0;line-height:1.7;">{line}</p>'
        )

    _close_ol()
    content_html = "\n".join(html_parts)

    # ── Google Doc button (shown at top when available) ───────
    doc_button = ""
    if doc_url:
        doc_button = f"""
        <div style="text-align:center;margin:20px 0 28px;">
          <a href="{doc_url}"
             style="display:inline-block;background:#1a73e8;color:white;
                    text-decoration:none;font-weight:bold;font-size:14px;
                    padding:12px 28px;border-radius:6px;">
            &#128196;&nbsp; View Full Report in Google Docs &rarr;
          </a>
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width"></head>
<body style="font-family:Arial,Helvetica,sans-serif;max-width:680px;
             margin:0 auto;background:#f4f4f4;color:#333;">

  <!-- Header banner -->
  <div style="background:#1a1a2e;color:white;padding:28px 32px;
              border-radius:8px 8px 0 0;">
    <h1 style="margin:0;font-size:22px;font-weight:700;">
      &#128202;&nbsp;{product} &mdash; Weekly Review Pulse
    </h1>
    <p style="margin:6px 0 0;opacity:0.75;font-size:14px;">Week of {week_start}</p>
  </div>

  <!-- Body card -->
  <div style="background:white;padding:32px 36px;
              border-radius:0 0 8px 8px;
              box-shadow:0 2px 8px rgba(0,0,0,0.08);">

    {doc_button}
    {content_html}

  </div>

  <!-- Footer -->
  <p style="text-align:center;font-size:11px;color:#aaa;margin:16px 0 24px;">
    Generated automatically by the AI Review Pulse Pipeline
  </p>

</body>
</html>"""


# ── MCP-based delivery ──────────────────────────────────────


def _create_google_doc_via_rest(
    report_markdown: str, product: str, week_start: str
) -> str | None:
    """Append the report to a Google Doc via ``POST /append_to_doc``.

    Returns the Doc URL on success, or None if the server is unavailable.
    """
    if not MCP_SERVER_URL:
        logger.warning(
            "MCP_SERVER_URL not configured — skipping Google Doc creation."
        )
        return None

    try:
        title = f"{product} — Weekly Pulse — {week_start}"
        result = append_to_doc(title=title, content=report_markdown)

        if result.get("success"):
            data = result.get("data")
            # Extract doc_url from the response
            if isinstance(data, dict):
                doc_url = data.get("doc_url") or data.get("url") or data.get("documentUrl")
            elif isinstance(data, str):
                # Server may return just the URL as a string
                doc_url = data if data.startswith("http") else None
            else:
                doc_url = None

            if doc_url:
                logger.info("Google Doc created via REST: %s", doc_url)
                return doc_url
            else:
                logger.warning(
                    "POST /append_to_doc returned success but no URL. "
                    "Response data: %s",
                    data,
                )
                return None

        logger.error("POST /append_to_doc failed: %s", result)
        return None

    except Exception as exc:
        logger.error("REST Google Doc creation failed: %s", exc)
        return None


def _send_email_via_rest(
    report_markdown: str,
    product: str,
    week_start: str,
    doc_url: str | None,
) -> bool:
    """Create an email draft via ``POST /create_email_draft``.

    Returns True on success, False otherwise.
    """
    if not MCP_SERVER_URL:
        logger.warning(
            "MCP_SERVER_URL not configured — skipping email delivery."
        )
        return False

    if not REPORT_RECIPIENTS:
        logger.info("No REPORT_RECIPIENTS configured — skipping email.")
        return False

    try:
        subject = f"{product} — Weekly Review Pulse — {week_start}"

        # Plain-text fallback (doc link appended at bottom)
        plain_body = report_markdown
        if doc_url:
            plain_body += f"\n\n---\nView full report: {doc_url}\n"

        # Rich HTML version
        html_body = _build_html_email(
            report_markdown, product, week_start, doc_url
        )

        result = create_email_draft(
            to=REPORT_RECIPIENTS,
            subject=subject,
            body=plain_body,
            body_html=html_body,
        )

        if result.get("success"):
            logger.info(
                "Email draft created via REST for: %s", ", ".join(REPORT_RECIPIENTS)
            )
            return True

        logger.error("POST /create_email_draft failed: %s", result)
        return False

    except Exception as exc:
        logger.error("REST email delivery failed: %s", exc)
        return False


# ── Main node ────────────────────────────────────────────────


def deliver(state: PipelineState) -> dict:
    """LangGraph node: deliver the validated report and write audit log.

    **Delivery channels** (via Railway-deployed FastAPI server — plain REST):

    1. Google Docs — ``POST /append_to_doc``
    2. Gmail — ``POST /create_email_draft``
    3. Local file + audit log — always written

    **Input** (from state):
        - ``report_markdown``    — validated report
        - ``cleaned_reviews``    — for review count in audit
        - ``clusters``           — for theme count in audit
        - ``product``            — product name
        - ``week_start``         — ISO date
        - ``week_end``           — ISO date
        - ``validation_passed``  — whether validation passed
        - ``validation_errors``  — any validation warnings

    **Output**::

        {
            "doc_url": str | None,
            "email_sent": bool,
            "audit_record": dict
        }
    """
    report_markdown = state.get("report_markdown", "")
    fee_explainer = state.get("fee_explainer", "")
    fee_pain_point = state.get("fee_pain_point", "")
    cleaned_reviews = state.get("cleaned_reviews", [])
    clusters = state.get("clusters", [])
    product = state.get("product", "Groww")
    week_start = state.get("week_start", "")
    week_end = state.get("week_end", "")
    validation_passed = state.get("validation_passed", False)
    validation_errors = state.get("validation_errors", [])

    # ── Combine report + fee explainer for delivery ──────────
    full_content = report_markdown
    if fee_explainer:
        full_content += (
            f"\n\n---\n\n"
            f"## Fee Explainer\n\n"
            f"**User confusion:** {fee_pain_point}\n\n"
            f"{fee_explainer}"
        )

    # ── Check for existing delivery (idempotency) ────────────
    existing_audit = _load_audit_log(product, week_start)
    if existing_audit and existing_audit.get("email_sent"):
        logger.info(
            "Delivery already completed for %s week %s — skipping.",
            product,
            week_start,
        )
        return {
            "doc_url": existing_audit.get("doc_url"),
            "email_sent": existing_audit.get("email_sent", False),
            "audit_record": existing_audit,
        }

    # ── Save report locally (always) ─────────────────────────
    report_path = _save_report_locally(full_content, product, week_start)

    # ── Google Docs (via REST /append_to_doc) ───────────────
    doc_url = _create_google_doc_via_rest(full_content, product, week_start)

    # ── Gmail (via REST /create_email_draft) ─────────────────
    email_sent = _send_email_via_rest(
        full_content, product, week_start, doc_url
    )

    # ── Audit log ────────────────────────────────────────────
    audit_record = {
        "product": product,
        "week_start": week_start,
        "week_end": week_end,
        "review_count": len(cleaned_reviews),
        "theme_count": len(clusters),
        "report_hash": _compute_report_hash(full_content),
        "report_path": report_path,
        "doc_url": doc_url,
        "email_sent": email_sent,
        "validation_passed": validation_passed,
        "validation_errors": validation_errors,
        "fee_pain_point": fee_pain_point,
        "has_fee_explainer": bool(fee_explainer),
        "delivery_method": "rest_api",
        "server_url": MCP_SERVER_URL or "(not configured)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _save_audit_log(audit_record, product, week_start)

    logger.info(
        "Delivery complete — Doc: %s | Email: %s | Audit logged.",
        doc_url or "(local only)",
        email_sent,
    )

    return {
        "doc_url": doc_url,
        "email_sent": email_sent,
        "audit_record": audit_record,
    }
