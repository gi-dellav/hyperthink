"""
mcp.py — MCP (Model Context Protocol) client for HyperThink.

Wraps an MCP stdio server and exposes a sync interface compatible with
HyperThink's tool system (LiteLLM schemas + executor callables).

The MCP SDK is async; this module bridges it via a background thread with
its own event loop. The MCP session is kept alive inside that thread until
``close()`` is called.

Usage
-----
    from hyperthink_litellm.tools.mcp import MCPClient

    with MCPClient("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/"]) as c:
        print(c.get_tools())

Requires the optional ``mcp`` dependency::

    pip install 'hyperthink-litellm[mcp]'
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Callable

# mcp is an optional dependency (hyperthink-litellm[mcp]).
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False


def _convert_tool(mcp_tool) -> dict:
    """Convert an MCP tool definition to a LiteLLM-compatible schema."""
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description or "",
            "parameters": mcp_tool.inputSchema,
        },
    }


class MCPClient:
    """
    Synchronous wrapper around an MCP stdio server.

    Keeps the MCP session alive in a background thread so callers can
    use it from synchronous code without managing an event loop.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict | None = None,
    ) -> None:
        if not _MCP_AVAILABLE:
            raise ImportError(
                "The 'mcp' package is required to use MCPClient. "
                "Install it with: pip install 'hyperthink-litellm[mcp]'"
            )
        self._command = command
        self._args = args
        self._env = env

        self._tools: list[dict] = []
        self._session: ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._done: asyncio.Event | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Start the background thread and initialise the MCP session."""
        started = threading.Event()
        error_box: list[Exception | None] = [None]

        def _target() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._done = asyncio.Event()
            loop.run_until_complete(self._main(started, error_box))

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()
        started.wait(timeout=30)
        if error_box[0] is not None:
            raise error_box[0]

    def close(self) -> None:
        """Signal the background thread to shut down and wait for it."""
        if self._loop is not None and self._done is not None:
            self._loop.call_soon_threadsafe(self._done.set)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def get_tools(self) -> list[dict]:
        """Return LiteLLM-compatible tool schemas for all tools on this server."""
        return list(self._tools)

    def get_executors(self) -> dict[str, Callable[[str], str]]:
        """
        Return a name→executor mapping for all tools on this server.

        Each executor accepts a JSON string of arguments and returns a
        plain-text result string.
        """
        return {
            schema["function"]["name"]: self._make_executor(schema["function"]["name"])
            for schema in self._tools
        }

    def _make_executor(self, name: str) -> Callable[[str], str]:
        def executor(arguments: str) -> str:
            return self._call_tool_sync(name, arguments)
        executor.__name__ = name
        return executor

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "MCPClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Async internals (run inside the background thread's event loop)
    # ------------------------------------------------------------------

    async def _main(
        self,
        started: threading.Event,
        error_box: list[Exception | None],
    ) -> None:
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    self._session = session
                    await session.initialize()
                    tools_result = await session.list_tools()
                    self._tools = [_convert_tool(t) for t in tools_result.tools]
                    started.set()
                    # Keep the session alive until close() is called.
                    await self._done.wait()
        except Exception as exc:
            error_box[0] = exc
            started.set()
        finally:
            self._session = None

    def _call_tool_sync(self, name: str, arguments: str) -> str:
        """Call a tool on the MCP server and return the text result."""
        if self._session is None or self._loop is None:
            return "Error: MCP session is not connected."
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return f"Error: Could not parse tool arguments as JSON: {exc}"

        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, parsed),
            self._loop,
        )
        try:
            result = future.result(timeout=60)
        except TimeoutError:
            return f"Error: Tool call '{name}' timed out after 60 seconds."
        except Exception as exc:
            return f"Error calling tool '{name}': {exc}"

        # MCP CallToolResult has a .content list; extract text items.
        parts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(parts) if parts else str(result)
