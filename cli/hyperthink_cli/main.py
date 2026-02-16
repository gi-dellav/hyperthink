"""
main.py — HyperThink CLI chat agent.

Modes:
    ASK   — direct LiteLLM inference with live streaming
    SOLVE — HyperThink dual-model scaffolding call

Commands:
    /clear               clear the terminal and reset conversation context
    /mode ask|solve      switch between modes
    /apikey <key>        set the OpenRouter API key for this session
    /help                show available commands

API key:
    Set OPENROUTER_API_KEY in the environment before starting, or use the
    /apikey command to set it interactively during the session.
"""

import os
import sys

import litellm
from rich.console import Console
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.completion import WordCompleter

# Support running directly from the repo without installing the package.
_LIB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "lib-litellm")
)
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)

from hyperthink_litellm import HyperThink  # noqa: E402
from hyperthink_litellm.defaults import DEFAULT_MODEL_A, DEFAULT_MODEL_B  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

MODE_ASK = "ASK"
MODE_SOLVE = "SOLVE"

_COMMANDS = ["/clear", "/mode ask", "/mode solve", "/apikey", "/help"]

_OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"

console = Console()


# ── HyperThink subclass with rich logging ─────────────────────────────────────


class _RichHyperThink(HyperThink):
    """HyperThink that routes _log() through the rich console."""

    def _log(self, msg: str) -> None:
        if self.logging_enabled:
            console.log(f"[dim]{msg}[/dim]")


# ── Prompt helper ─────────────────────────────────────────────────────────────


def _prompt_text(mode: str) -> HTML:
    if mode == MODE_ASK:
        return HTML("<ansicyan><b>[ASK]</b></ansicyan> <ansiwhite>›</ansiwhite> ")
    return HTML("<ansimagenta><b>[SOLVE]</b></ansimagenta> <ansiwhite>›</ansiwhite> ")


# ── Inference helpers ─────────────────────────────────────────────────────────


def _run_ask(messages: list, model: str) -> str:
    """Stream a direct LiteLLM inference; return the full response text."""
    response = litellm.completion(model=model, messages=messages, stream=True)
    full_text = ""
    console.print()
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            full_text += delta
            print(delta, end="", flush=True)
    print()
    console.print()
    return full_text


def _run_solve(messages: list, model_a: str, model_b: str) -> str:
    """Run a HyperThink scaffolding query; return the final answer."""
    ht = _RichHyperThink(
        model_a=model_a,
        model_b=model_b,
        logging_enabled=True,
    )
    console.print()
    result = ht.query(messages)
    console.print()
    console.print(Markdown(result))
    console.print()
    return result


# ── Main REPL ─────────────────────────────────────────────────────────────────


def main() -> None:
    model_a = os.environ.get("HYPERTHINK_MODEL_A", DEFAULT_MODEL_A)
    model_b = os.environ.get("HYPERTHINK_MODEL_B", DEFAULT_MODEL_B)
    system_prompt = os.environ.get("HYPERTHINK_SYSTEM", "You are a helpful assistant.")

    mode = MODE_ASK
    history: list[dict] = []

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        completer=WordCompleter(_COMMANDS, sentence=True),
        complete_while_typing=False,
    )

    # ── Banner ────────────────────────────────────────────────────────────────
    console.rule("[bold]HyperThink CLI[/bold]")
    console.print(
        f"  Model A [cyan]{model_a}[/cyan]  ·  Model B [cyan]{model_b}[/cyan]"
    )
    _api_key = os.environ.get(_OPENROUTER_KEY_ENV, "")
    if _api_key:
        _masked = _api_key[:6] + "…" + _api_key[-4:]
        console.print(f"  API key [green]{_masked}[/green] (from environment)")
    else:
        console.print(
            f"  [yellow]No {_OPENROUTER_KEY_ENV} set.[/yellow] "
            "Use [bold]/apikey <key>[/bold] to set it."
        )
    console.print(
        "  [dim]/mode ask[/dim]  [dim]/mode solve[/dim]  "
        "[dim]/apikey[/dim]  [dim]/clear[/dim]  [dim]/help[/dim]"
    )
    console.rule()
    console.print()

    # ── REPL loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw = session.prompt(_prompt_text(mode))
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print("[dim]Goodbye.[/dim]")
            break

        user_input = raw.strip()
        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip().lower() if len(parts) > 1 else ""

            if cmd == "/clear":
                history.clear()
                os.system("clear")
                continue

            if cmd == "/mode":
                if not arg:
                    console.print(
                        f"Current mode: [bold]{mode}[/bold]. "
                        "Usage: [yellow]/mode ask[/yellow] | "
                        "[yellow]/mode solve[/yellow]"
                    )
                    continue
                new_mode = arg.upper()
                if new_mode not in (MODE_ASK, MODE_SOLVE):
                    console.print(
                        f"[red]Unknown mode:[/red] '{arg}'. "
                        "Use [yellow]ask[/yellow] or [yellow]solve[/yellow]."
                    )
                    continue
                if new_mode == mode:
                    console.print(f"Already in [bold]{mode}[/bold] mode.")
                else:
                    mode = new_mode
                    console.print(f"[green]Switched to {mode} mode.[/green]")
                continue

            if cmd == "/apikey":
                arg_raw = parts[1].strip() if len(parts) > 1 else ""
                if not arg_raw:
                    current = os.environ.get(_OPENROUTER_KEY_ENV, "")
                    if current:
                        _masked = current[:6] + "…" + current[-4:]
                        console.print(
                            f"Current API key: [green]{_masked}[/green]"
                        )
                    else:
                        console.print(
                            f"[yellow]No API key set.[/yellow] "
                            f"Usage: [bold]/apikey <key>[/bold]"
                        )
                else:
                    os.environ[_OPENROUTER_KEY_ENV] = arg_raw
                    _masked = arg_raw[:6] + "…" + arg_raw[-4:]
                    console.print(f"[green]API key updated:[/green] {_masked}")
                continue

            if cmd == "/help":
                console.print()
                console.print("[bold]Commands[/bold]")
                console.print(
                    "  [yellow]/mode ask[/yellow]     "
                    "direct LLM inference with streaming"
                )
                console.print(
                    "  [yellow]/mode solve[/yellow]   "
                    "HyperThink dual-model scaffolding"
                )
                console.print(
                    "  [yellow]/apikey <key>[/yellow] "
                    "set the OpenRouter API key for this session"
                )
                console.print(
                    "  [yellow]/clear[/yellow]        "
                    "clear terminal and reset conversation context"
                )
                console.print("  [yellow]/help[/yellow]         " "show this message")
                console.print()
                continue

            console.print(
                f"[red]Unknown command:[/red] {cmd}. "
                "Type [yellow]/help[/yellow] for available commands."
            )
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        try:
            if mode == MODE_ASK:
                msgs = [{"role": "system", "content": system_prompt}, *history]
                answer = _run_ask(msgs, model_a)
            else:
                answer = _run_solve(list(history), model_a, model_b)
            history.append({"role": "assistant", "content": answer})
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            if history and history[-1]["role"] == "user":
                history.pop()
        except Exception as exc:
            console.print(f"\n[red]Error:[/red] {exc}")
            if history and history[-1]["role"] == "user":
                history.pop()


if __name__ == "__main__":
    main()
