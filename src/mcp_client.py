"""
MCP client utility for communicating with the Railway-deployed MCP server.

Provides async helper functions to connect via SSE and invoke MCP tools
(``create_google_doc``, ``send_email``) on the remote server.

The MCP server has OAuth credentials and token configuration embedded,
so the client only needs the server URL to connect.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.sse import sse_client

from src.config import MCP_SERVER_URL, MCP_SERVER_TIMEOUT

logger = logging.getLogger(__name__)


# ── Session management ───────────────────────────────────────


@asynccontextmanager
async def get_mcp_session():
    """Create an MCP client session connected to the Railway server via SSE.

    Usage::

        async with get_mcp_session() as session:
            result = await session.call_tool("send_email", {...})
    """
    if not MCP_SERVER_URL:
        raise RuntimeError(
            "MCP_SERVER_URL is not configured. "
            "Set it in .env to point at your Railway MCP server."
        )

    logger.info("Connecting to MCP server at %s", MCP_SERVER_URL)

    async with sse_client(url=MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("MCP session initialised successfully.")
            yield session


# ── Tool invocation ──────────────────────────────────────────


async def _call_tool_async(tool_name: str, arguments: dict) -> dict:
    """Call a named tool on the MCP server and return parsed result.

    Returns a dict with ``success`` (bool) and ``data`` (parsed JSON or
    raw text from the first text content block).
    """
    async with get_mcp_session() as session:
        logger.info("Calling MCP tool '%s' with args: %s", tool_name, arguments)
        result = await session.call_tool(tool_name, arguments=arguments)

        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    # Try to parse as JSON; fall back to raw text
                    try:
                        parsed = json.loads(block.text)
                        return {"success": True, "data": parsed}
                    except (json.JSONDecodeError, TypeError):
                        return {"success": True, "data": block.text}

        return {"success": True, "data": None}


def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Synchronous wrapper around :func:`_call_tool_async`.

    Safe to call from synchronous LangGraph node functions.  Creates a
    new event loop if one is not already running.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an existing async context (e.g. Jupyter, some
        # LangGraph runtimes).  Use nest_asyncio or a thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _call_tool_async(tool_name, arguments)).result(
                timeout=MCP_SERVER_TIMEOUT
            )
    else:
        return asyncio.run(_call_tool_async(tool_name, arguments))


# ── Discovery ────────────────────────────────────────────────


async def _list_tools_async() -> list[dict]:
    """List all tools available on the MCP server."""
    async with get_mcp_session() as session:
        tools_response = await session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.inputSchema if hasattr(t, "inputSchema") else None,
            }
            for t in tools_response.tools
        ]


def list_mcp_tools() -> list[dict]:
    """Synchronous wrapper to list all tools on the MCP server.

    Useful for debugging and verifying the server is reachable::

        from src.mcp_client import list_mcp_tools
        tools = list_mcp_tools()
        for t in tools:
            print(f"  {t['name']}: {t['description']}")
    """
    return asyncio.run(_list_tools_async())
