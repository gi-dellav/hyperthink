"""
tools — LLM-callable tool definitions for HyperThink.

Currently available:
  math_tools   — SymPy-powered math solver (solve, integrate, differentiate,
                  simplify, limit, expand, factor, series, evaluate, latex).
  MCPClient    — Connect to any MCP stdio server and expose its tools.
"""

from .math import MATH_TOOLS, execute_math_tool
from .mcp import MCPClient

__all__ = ["MATH_TOOLS", "execute_math_tool", "MCPClient"]
