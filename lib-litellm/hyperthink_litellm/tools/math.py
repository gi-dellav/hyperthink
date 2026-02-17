"""
math.py — SymPy-powered math solver tool for LLMs.

Provides:
  MATH_TOOLS        — LiteLLM/OpenAI-compatible tool schema list.
  execute_math_tool — Dispatch function that runs a math_solver tool call.

Supported operations
--------------------
  solve          Solve an equation or system of equations.
  integrate      Indefinite or definite integral.
  differentiate  Derivative of any order.
  simplify       Algebraic simplification.
  limit          Limit of an expression.
  expand         Expand an algebraic expression.
  factor         Factor a polynomial.
  series         Taylor/Laurent series expansion.
  evaluate       Numerical evaluation (arbitrary precision).
  latex          Render expression as LaTeX.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Dict, List, Optional

# SymPy is an optional dependency (hyperthink-litellm[math]).
try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Tool JSON schema
# ---------------------------------------------------------------------------

MATH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "math_solver",
        "description": (
            "Evaluate symbolic mathematics using SymPy. "
            "Use this tool whenever exact computation, symbolic simplification, "
            "or verification of a mathematical result is needed. "
            "Supported operations: solve, integrate, differentiate, simplify, "
            "limit, expand, factor, series, evaluate, latex."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [
                        "solve",
                        "integrate",
                        "differentiate",
                        "simplify",
                        "limit",
                        "expand",
                        "factor",
                        "series",
                        "evaluate",
                        "latex",
                    ],
                    "description": "The mathematical operation to perform.",
                },
                "expression": {
                    "type": "string",
                    "description": (
                        "Primary mathematical expression using Python/SymPy syntax. "
                        "Use ** for powers (x**2), * for multiplication, "
                        "sqrt(x), sin(x), cos(x), exp(x), log(x), pi, E, oo (infinity). "
                        "For equations write 'lhs = rhs' or just 'lhs' (implying = 0). "
                        "Required for all operations except when 'equations' is used."
                    ),
                },
                "variable": {
                    "type": "string",
                    "description": (
                        "Variable name(s) to operate on, default 'x'. "
                        "Comma-separate for multiple variables, e.g. 'x,y,z'."
                    ),
                },
                "equations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "List of equations for solving a system. Each entry is a string "
                        "like 'x + y = 3' or 'x**2 + y**2 - 1'. "
                        "Use instead of 'expression' when solving multiple equations."
                    ),
                },
                "lower_bound": {
                    "type": "string",
                    "description": "Lower bound for definite integrals (SymPy expression string).",
                },
                "upper_bound": {
                    "type": "string",
                    "description": "Upper bound for definite integrals (SymPy expression string).",
                },
                "point": {
                    "type": "string",
                    "description": (
                        "Point for limit evaluation or series expansion centre. "
                        "Use 'oo' for +∞ or '-oo' for -∞."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["+", "-", "+-"],
                    "description": (
                        "Direction for limit evaluation: '+' (right), '-' (left), "
                        "or '+-' (two-sided, default)."
                    ),
                },
                "order": {
                    "type": "integer",
                    "description": (
                        "Order of differentiation (default 1) "
                        "or number of terms in a series expansion (default 6)."
                    ),
                },
            },
            "required": ["operation"],
        },
    },
}

#: Ready-to-use list to pass as the ``tools`` argument to LiteLLM / OpenAI.
MATH_TOOLS: List[Dict[str, Any]] = [MATH_TOOL_SCHEMA]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRANSFORMATIONS = None  # lazily initialised


def _get_transformations():
    global _TRANSFORMATIONS
    if _TRANSFORMATIONS is None:
        _TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)
    return _TRANSFORMATIONS


def _build_namespace(var_names: List[str]) -> Dict[str, Any]:
    """Return a SymPy-only namespace (no Python builtins) plus requested symbols."""
    ns: Dict[str, Any] = {
        k: v
        for k, v in sp.__dict__.items()
        if not k.startswith("_") and isinstance(v, (type, sp.Basic.__class__))
    }
    # Always include commonly used objects explicitly
    ns.update(
        {
            "pi": sp.pi,
            "E": sp.E,
            "oo": sp.oo,
            "zoo": sp.zoo,
            "nan": sp.nan,
            "I": sp.I,
            "sqrt": sp.sqrt,
            "cbrt": lambda x: sp.cbrt(x),
            "exp": sp.exp,
            "log": sp.log,
            "ln": sp.log,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "atan2": sp.atan2,
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "asinh": sp.asinh,
            "acosh": sp.acosh,
            "atanh": sp.atanh,
            "Abs": sp.Abs,
            "sign": sp.sign,
            "floor": sp.floor,
            "ceiling": sp.ceiling,
            "factorial": sp.factorial,
            "binomial": sp.binomial,
            "gamma": sp.gamma,
            "beta": sp.beta,
            "erf": sp.erf,
            "erfc": sp.erfc,
            "conjugate": sp.conjugate,
            "re": sp.re,
            "im": sp.im,
            "Rational": sp.Rational,
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Matrix": sp.Matrix,
            "symbols": sp.symbols,
        }
    )
    for v in var_names:
        if v not in ns:
            ns[v] = sp.Symbol(v)
    return ns


def _parse(expr_str: str, ns: Dict[str, Any]) -> sp.Basic:
    """Parse an expression string into a SymPy expression."""
    return parse_expr(
        expr_str.strip(),
        local_dict=ns,
        global_dict={},
        transformations=_get_transformations(),
        evaluate=True,
    )


def _parse_equation(eq_str: str, ns: Dict[str, Any]) -> sp.Basic:
    """Parse 'lhs = rhs' into Eq(lhs, rhs), or 'lhs' into Eq(lhs, 0)."""
    eq_str = eq_str.strip()
    if "==" in eq_str:
        lhs, rhs = eq_str.split("==", 1)
        return sp.Eq(_parse(lhs, ns), _parse(rhs, ns))
    if "=" in eq_str:
        lhs, rhs = eq_str.split("=", 1)
        return sp.Eq(_parse(lhs, ns), _parse(rhs, ns))
    return _parse(eq_str, ns)


# ---------------------------------------------------------------------------
# Operation implementations
# ---------------------------------------------------------------------------

def _op_solve(args: Dict, ns: Dict, var_syms: List) -> str:
    equations_raw: Optional[List[str]] = args.get("equations")
    expression: str = args.get("expression", "")

    if equations_raw:
        eqs = [_parse_equation(e, ns) for e in equations_raw]
    elif expression:
        eqs = [_parse_equation(expression, ns)]
    else:
        return "Error: provide 'expression' or 'equations' for solve."

    solution = sp.solve(eqs, var_syms if len(var_syms) > 1 else var_syms[0])

    if not solution:
        return "No solution found."
    return f"Solution: {solution}"


def _op_integrate(args: Dict, ns: Dict, sym: sp.Symbol) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for integrate."
    expr = _parse(expression, ns)
    lower = args.get("lower_bound")
    upper = args.get("upper_bound")

    if lower is not None and upper is not None:
        lb = _parse(lower, ns)
        ub = _parse(upper, ns)
        result = sp.integrate(expr, (sym, lb, ub))
        result_simplified = sp.simplify(result)
        return (
            f"∫[{lower}, {upper}] ({expression}) d{sym} = {result_simplified}"
            + (f"\n  ≈ {sp.N(result_simplified, 15)}" if result_simplified.is_number else "")
        )
    else:
        result = sp.integrate(expr, sym)
        return f"∫ ({expression}) d{sym} = {result} + C"


def _op_differentiate(args: Dict, ns: Dict, sym: sp.Symbol) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for differentiate."
    expr = _parse(expression, ns)
    order: int = int(args.get("order", 1))
    result = sp.diff(expr, sym, order)
    result_simplified = sp.simplify(result)
    label = "d" if order == 1 else f"d^{order}"
    denom = f"d{sym}" if order == 1 else f"d{sym}^{order}"
    return f"{label}/{denom} ({expression}) = {result_simplified}"


def _op_simplify(args: Dict, ns: Dict) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for simplify."
    expr = _parse(expression, ns)
    result = sp.simplify(expr)
    return f"simplify({expression}) = {result}"


def _op_limit(args: Dict, ns: Dict, sym: sp.Symbol) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for limit."
    expr = _parse(expression, ns)
    point_str: str = args.get("point", "0")
    direction: str = args.get("direction", "+-")
    pt = _parse(point_str, ns)
    result = sp.limit(expr, sym, pt, direction)
    arrow = f"{point_str}{'⁺' if direction == '+' else '⁻' if direction == '-' else ''}"
    return f"lim({sym} → {arrow}) ({expression}) = {result}"


def _op_expand(args: Dict, ns: Dict) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for expand."
    expr = _parse(expression, ns)
    result = sp.expand(expr)
    return f"expand({expression}) = {result}"


def _op_factor(args: Dict, ns: Dict) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for factor."
    expr = _parse(expression, ns)
    result = sp.factor(expr)
    return f"factor({expression}) = {result}"


def _op_series(args: Dict, ns: Dict, sym: sp.Symbol) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for series."
    expr = _parse(expression, ns)
    point_str: str = args.get("point", "0")
    order: int = int(args.get("order", 6))
    pt = _parse(point_str, ns)
    result = sp.series(expr, sym, pt, order)
    return f"series({expression}, {sym}={point_str}, order={order}):\n  {result}"


def _op_evaluate(args: Dict, ns: Dict) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for evaluate."
    expr = _parse(expression, ns)
    result = sp.N(expr, 50)  # 50 significant digits
    return f"N({expression}) = {result}"


def _op_latex(args: Dict, ns: Dict) -> str:
    expression: str = args.get("expression", "")
    if not expression:
        return "Error: 'expression' is required for latex."
    expr = _parse(expression, ns)
    result = sp.latex(expr)
    return f"LaTeX: {result}"


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def execute_math_tool(arguments: str | Dict[str, Any]) -> str:
    """
    Execute a ``math_solver`` tool call.

    Parameters
    ----------
    arguments:
        Either the raw JSON string from the tool-call response or an already-
        decoded dict.

    Returns
    -------
    str
        Human-readable result string, or an error message starting with
        ``"Error: "`` if something went wrong.
    """
    if not _SYMPY_AVAILABLE:
        return (
            "Error: SymPy is not installed. "
            "Install it with: pip install 'hyperthink-litellm[math]'"
        )

    # Decode arguments if they arrive as a JSON string
    if isinstance(arguments, str):
        try:
            args: Dict[str, Any] = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return f"Error: Could not parse tool arguments as JSON: {exc}"
    else:
        args = arguments

    operation: str = args.get("operation", "")
    if not operation:
        return "Error: 'operation' is required."

    # Parse variable list
    variable_str: str = args.get("variable", "x")
    var_names: List[str] = [v.strip() for v in variable_str.split(",") if v.strip()]
    if not var_names:
        var_names = ["x"]

    # Build SymPy namespace and symbol list
    ns = _build_namespace(var_names)
    var_syms: List[sp.Symbol] = [ns[v] for v in var_names]
    primary_sym: sp.Symbol = var_syms[0]

    try:
        if operation == "solve":
            return _op_solve(args, ns, var_syms)
        elif operation == "integrate":
            return _op_integrate(args, ns, primary_sym)
        elif operation == "differentiate":
            return _op_differentiate(args, ns, primary_sym)
        elif operation == "simplify":
            return _op_simplify(args, ns)
        elif operation == "limit":
            return _op_limit(args, ns, primary_sym)
        elif operation == "expand":
            return _op_expand(args, ns)
        elif operation == "factor":
            return _op_factor(args, ns)
        elif operation == "series":
            return _op_series(args, ns, primary_sym)
        elif operation == "evaluate":
            return _op_evaluate(args, ns)
        elif operation == "latex":
            return _op_latex(args, ns)
        else:
            return (
                f"Error: Unknown operation '{operation}'. "
                "Valid options: solve, integrate, differentiate, simplify, "
                "limit, expand, factor, series, evaluate, latex."
            )
    except Exception:
        return f"Error executing math_solver({operation}):\n{traceback.format_exc()}"
