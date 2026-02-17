"""
tools — LLM-callable tool definitions for HyperThink.

Currently available:
  math_tools   — SymPy-powered math solver (solve, integrate, differentiate,
                  simplify, limit, expand, factor, series, evaluate, latex).
"""

from .math import MATH_TOOLS, execute_math_tool

__all__ = ["MATH_TOOLS", "execute_math_tool"]
