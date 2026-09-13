"""
REST client utility for communicating with the Railway-deployed FastAPI server.

Provides synchronous helper functions to call the server's REST endpoints:
- ``POST /upload_to_drive``    — upload the detailed PDF to Google Drive
- ``POST /append_to_doc``      — append report content to a Google Doc
- ``POST /create_email_draft`` — create a Gmail draft (or send directly)

The FastAPI server has OAuth credentials and token configuration embedded,
so the client only needs the server URL (``MCP_SERVER_URL``) to connect.
No SSE or MCP protocol overhead — plain JSON over HTTPS.
"""

import base64
import logging

import httpx

from src.config import MCP_SERVER_URL, MCP_SERVER_TIMEOUT

logger = logging.getLogger(__name__)


# ── Internal helpers ─────────────────────────────────────────


def _post(endpoint: str, payload: dict) -> dict:
    """Send a JSON POST request to *endpoint* on the configured server.

    Args:
        endpoint: Path relative to ``MCP_SERVER_URL``, e.g. ``/append_to_doc``.
        payload:  JSON-serialisable request body.

    Returns:
        A dict with ``success`` (bool) and ``data`` (response JSON or None).

    Raises:
        RuntimeError: If ``MCP_SERVER_URL`` is not configured.
        httpx.HTTPStatusError: On 4xx / 5xx responses (after logging).
        httpx.RequestError: On network-level failures (after logging).
    """
    if not MCP_SERVER_URL:
        raise RuntimeError(
            "MCP_SERVER_URL is not configured. "
            "Set it in .env to point at your Railway FastAPI server."
        )

    base = MCP_SERVER_URL.rstrip("/")
    url = f"{base}{endpoint}"

    logger.info("POST %s — payload keys: %s", url, list(payload.keys()))

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=MCP_SERVER_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Server returned HTTP %s for %s: %s",
            exc.response.status_code,
            url,
            exc.response.text[:200],
        )
        raise
    except httpx.RequestError as exc:
        logger.error("Network error calling %s: %s", url, exc)
        raise

    try:
        data = response.json()
    except Exception:
        # Non-JSON success body (e.g. plain 200 OK with empty body)
        data = response.text or None

    logger.info("POST %s → HTTP 2xx, data: %s", url, str(data)[:120])
    return {"success": True, "data": data}


# ── Public API ───────────────────────────────────────────────


def append_to_doc(title: str, content: str) -> dict:
    """Append *content* to a Google Doc (created if it doesn't exist).

    Calls ``POST /append_to_doc`` on the Railway FastAPI server.

    Args:
        title:   Document title (used to look up or create the Doc).
        content: Markdown/plain-text body to append.

    Returns:
        ``{"success": True, "data": {...}}`` where ``data`` typically
        contains ``doc_url`` and ``document_id``.

    Example::

        result = append_to_doc(
            title="Groww — Weekly Pulse — 2026-09-01",
            content="## Summary\\n...",
        )
        doc_url = result["data"]["doc_url"]
    """
    return _post("/append_to_doc", {"doc_id": "", "title": title, "content": content})


def upload_to_drive(
    filename: str,
    content: bytes,
    mime_type: str = "application/pdf",
) -> dict:
    """Upload a binary file to Google Drive and return its shareable URL.

    Calls ``POST /upload_to_drive`` on the Railway FastAPI server. The binary
    is base64-encoded for JSON transport; the server decodes it, uploads via
    the Drive API using its own OAuth credentials, and returns a link.

    Args:
        filename:  Destination filename, e.g. ``groww_pulse_2026-09-07.pdf``.
        content:   Raw file bytes.
        mime_type: MIME type of *content*.

    Returns:
        ``{"success": True, "data": {...}}`` where ``data`` typically
        contains ``file_url`` and ``file_id``.

    Example::

        result = upload_to_drive("pulse.pdf", pdf_bytes)
        url = result["data"]["file_url"]
    """
    payload = {
        "filename": filename,
        "mime_type": mime_type,
        "content_b64": base64.b64encode(content).decode("ascii"),
    }
    logger.info(
        "Uploading %s to Drive (%.1f KB before encoding)",
        filename,
        len(content) / 1024,
    )
    return _post("/upload_to_drive", payload)


def create_email_draft(
    to: list[str],
    subject: str,
    body: str,
    body_html: str | None = None,
) -> dict:
    """Create a Gmail draft (or send immediately, depending on server config).

    Calls ``POST /create_email_draft`` on the Railway FastAPI server.

    Args:
        to:        List of recipient email addresses.
        subject:   Email subject line.
        body:      Plain-text fallback email body.
        body_html: Optional HTML version of the body (rendered by email
                   clients that support HTML). Sent as ``body_html`` field
                   alongside ``body``.

    Returns:
        ``{"success": True, "data": {...}}`` where ``data`` typically
        contains ``draft_id`` or ``message_id``.

    Example::

        result = create_email_draft(
            to=["team@example.com"],
            subject="Weekly Pulse",
            body="## Summary\\n...",
            body_html="<h2>Summary</h2>...",
        )
    """
    # Server expects `to` as a plain string (comma-separated for multiple recipients)
    to_str = ", ".join(to) if isinstance(to, list) else to
    payload: dict = {"to": to_str, "subject": subject, "body": body}
    if body_html:
        payload["body_html"] = body_html
    return _post("/create_email_draft", payload)


# ── Legacy shim (kept for backward-compat during transition) ─


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """**Deprecated shim** — routes legacy MCP tool calls to REST endpoints.

    Supported mappings:

    - ``create_google_doc``  → :func:`append_to_doc`
    - ``send_email``         → :func:`create_email_draft`

    Raises:
        ValueError: For unknown tool names.
    """
    logger.warning(
        "call_mcp_tool('%s') is deprecated — use append_to_doc() or "
        "create_email_draft() directly.",
        tool_name,
    )

    if tool_name == "create_google_doc":
        return append_to_doc(
            title=arguments.get("title", "Untitled"),
            content=arguments.get("content", ""),
        )
    elif tool_name == "send_email":
        recipients = arguments.get("to", [])
        if isinstance(recipients, str):
            recipients = [recipients]
        return create_email_draft(
            to=recipients,
            subject=arguments.get("subject", ""),
            body=arguments.get("body", ""),
        )
    else:
        raise ValueError(
            f"Unknown tool '{tool_name}'. "
            "Use append_to_doc() or create_email_draft() directly."
        )
