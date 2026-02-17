"""
main.py — HyperThink CLI chat agent.

Modes:
    ASK   — direct LiteLLM inference with live streaming
    SOLVE — HyperThink dual-model scaffolding call
    PLAN  — decompose query into subtasks, solve each with HyperThink, synthesize

Commands:
    /clear                           clear the terminal and reset conversation context
    /mode ask|solve|plan             switch between modes
    /apikey <key>                    set the OpenRouter API key for this session
    /help                            show available commands
    /load <path>                     load a file's contents into the conversation context
    /reasoning-effort [a|b] <level>  set reasoning effort (low/medium/high/none)
    /mcp <command> [args...]         connect to an MCP stdio server
    /mcp disconnect                  close all active MCP connections

API key:
    Set OPENROUTER_API_KEY in the environment before starting, or use the
    /apikey command to set it interactively during the session.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*Pydantic serializer.*")

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

from hyperthink_litellm.defaults import DEFAULT_MODEL_A, DEFAULT_MODEL_B  # noqa: E402
from hyperthink_litellm.tools.mcp import MCPClient, _MCP_AVAILABLE  # noqa: E402

from .constants import (
    MODE_ASK,
    MODE_PLAN,
    MODE_SOLVE,
    _COMMANDS,
    _OPENROUTER_KEY_ENV,
    console,
)  # noqa: E402
from .inference import _run_ask, _run_plan, _run_solve  # noqa: E402


# ── Prompt helper ─────────────────────────────────────────────────────────────


def _prompt_text(mode: str) -> HTML:
    if mode == MODE_ASK:
        return HTML("<ansicyan><b>[ASK]</b></ansicyan> <ansiwhite>›</ansiwhite> ")
    if mode == MODE_PLAN:
        return HTML("<ansiyellow><b>[PLAN]</b></ansiyellow> <ansiwhite>›</ansiwhite> ")
    return HTML("<ansimagenta><b>[SOLVE]</b></ansimagenta> <ansiwhite>›</ansiwhite> ")


# ── Main REPL ─────────────────────────────────────────────────────────────────


def main() -> None:
    model_a = os.environ.get("HYPERTHINK_MODEL_A", DEFAULT_MODEL_A)
    model_b = os.environ.get("HYPERTHINK_MODEL_B", DEFAULT_MODEL_B)
    system_prompt = os.environ.get("HYPERTHINK_SYSTEM", "You are a helpful assistant.")

    mode = MODE_SOLVE
    history: list[dict] = []
    reasoning_effort_a: str | None = None
    reasoning_effort_b: str | None = None
    mcp_clients: list[MCPClient] = []

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
        "  [dim]/mode ask[/dim]  [dim]/mode solve[/dim]  [dim]/mode plan[/dim]  "
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
            for c in mcp_clients:
                c.close()
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
                        "[yellow]/mode solve[/yellow] | "
                        "[yellow]/mode plan[/yellow]"
                    )
                    continue
                new_mode = arg.upper()
                if new_mode not in (MODE_ASK, MODE_SOLVE, MODE_PLAN):
                    console.print(
                        f"[red]Unknown mode:[/red] '{arg}'. "
                        "Use [yellow]ask[/yellow], [yellow]solve[/yellow], "
                        "or [yellow]plan[/yellow]."
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
                        console.print(f"Current API key: [green]{_masked}[/green]")
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
                    "  [yellow]/mode plan[/yellow]    "
                    "decompose query into subtasks, solve each, synthesize"
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
                console.print(
                    "  [yellow]/load <path>[/yellow]   "
                    "load a file into the conversation context"
                )
                console.print(
                    "  [yellow]/reasoning-effort [a|b] <level>[/yellow]  "
                    "set reasoning effort (low · medium · high · none)"
                )
                console.print(
                    "  [yellow]/mcp <command> [args...][/yellow]  "
                    "connect to an MCP stdio server and load its tools"
                )
                console.print(
                    "  [yellow]/mcp disconnect[/yellow]  "
                    "close all active MCP connections"
                )
                console.print()
                continue

            if cmd == "/reasoning-effort":
                _VALID_EFFORTS = ("low", "medium", "high", "none")
                re_parts = (parts[1].split() if len(parts) > 1 else [])
                if not re_parts:
                    a_display = reasoning_effort_a or "none"
                    b_display = reasoning_effort_b or "none"
                    console.print(
                        f"Reasoning effort — Model A: [cyan]{a_display}[/cyan]  "
                        f"Model B: [cyan]{b_display}[/cyan]"
                    )
                    console.print(
                        "Usage: [yellow]/reasoning-effort <level>[/yellow]  "
                        "or [yellow]/reasoning-effort a|b <level>[/yellow]  "
                        "(levels: low · medium · high · none)"
                    )
                    continue
                if len(re_parts) == 1:
                    level = re_parts[0].lower()
                    if level not in _VALID_EFFORTS:
                        console.print(
                            f"[red]Unknown level:[/red] '{level}'. "
                            "Choose from: low · medium · high · none"
                        )
                        continue
                    reasoning_effort_a = None if level == "none" else level
                    reasoning_effort_b = None if level == "none" else level
                    console.print(
                        f"[green]Reasoning effort set to[/green] [cyan]{level}[/cyan] "
                        "for both models."
                    )
                elif len(re_parts) == 2:
                    target, level = re_parts[0].lower(), re_parts[1].lower()
                    if target not in ("a", "b"):
                        console.print(
                            f"[red]Unknown target:[/red] '{target}'. Use 'a' or 'b'."
                        )
                        continue
                    if level not in _VALID_EFFORTS:
                        console.print(
                            f"[red]Unknown level:[/red] '{level}'. "
                            "Choose from: low · medium · high · none"
                        )
                        continue
                    value = None if level == "none" else level
                    if target == "a":
                        reasoning_effort_a = value
                        console.print(
                            f"[green]Model A reasoning effort:[/green] [cyan]{level}[/cyan]"
                        )
                    else:
                        reasoning_effort_b = value
                        console.print(
                            f"[green]Model B reasoning effort:[/green] [cyan]{level}[/cyan]"
                        )
                else:
                    console.print(
                        "Usage: [yellow]/reasoning-effort <level>[/yellow]  "
                        "or [yellow]/reasoning-effort a|b <level>[/yellow]"
                    )
                continue

            if cmd == "/load":
                arg_raw = parts[1].strip() if len(parts) > 1 else ""
                if not arg_raw:
                    console.print("Usage: [yellow]/load <filepath>[/yellow]")
                    continue
                path = os.path.expanduser(arg_raw)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        content = fh.read()
                except FileNotFoundError:
                    console.print(f"[red]File not found:[/red] {arg_raw}")
                    continue
                except OSError as exc:
                    console.print(f"[red]Cannot read file:[/red] {exc}")
                    continue
                history.append(
                    {
                        "role": "user",
                        "content": f"[File loaded: {arg_raw}]\n\n{content}",
                    }
                )
                history.append(
                    {
                        "role": "assistant",
                        "content": f"File `{arg_raw}` loaded into context.",
                    }
                )
                console.print(
                    f"[green]Loaded:[/green] {arg_raw} ({len(content)} chars)"
                )
                continue

            if cmd == "/mcp":
                if not _MCP_AVAILABLE:
                    console.print(
                        "[red]MCP support not installed.[/red] "
                        "Run: [bold]pip install 'hyperthink-litellm[mcp]'[/bold]"
                    )
                    continue
                mcp_arg_raw = parts[1].strip() if len(parts) > 1 else ""
                if not mcp_arg_raw:
                    if mcp_clients:
                        console.print(
                            f"[bold]{len(mcp_clients)}[/bold] MCP server(s) connected. "
                            "Use [yellow]/mcp disconnect[/yellow] to close all."
                        )
                    else:
                        console.print(
                            "Usage: [yellow]/mcp <command> [args...][/yellow]  "
                            "— connect to an MCP stdio server"
                        )
                    continue
                mcp_parts = mcp_arg_raw.split()
                if mcp_parts[0].lower() == "disconnect":
                    if not mcp_clients:
                        console.print("No MCP servers connected.")
                    else:
                        for c in mcp_clients:
                            c.close()
                        mcp_clients.clear()
                        console.print("[green]All MCP connections closed.[/green]")
                    continue
                mcp_command = mcp_parts[0]
                mcp_args = mcp_parts[1:]
                console.print(
                    f"Connecting to MCP server: [cyan]{mcp_command}[/cyan] "
                    + " ".join(mcp_args)
                )
                try:
                    client = MCPClient(mcp_command, mcp_args)
                    client.connect()
                    mcp_clients.append(client)
                    tool_names = [t["function"]["name"] for t in client.get_tools()]
                    console.print(
                        f"[green]Connected.[/green] "
                        f"Loaded [bold]{len(tool_names)}[/bold] tool(s): "
                        + ", ".join(f"[cyan]{n}[/cyan]" for n in tool_names)
                    )
                except Exception as exc:
                    console.print(f"[red]Failed to connect:[/red] {exc}")
                continue

            console.print(
                f"[red]Unknown command:[/red] {cmd}. "
                "Type [yellow]/help[/yellow] for available commands."
            )
            continue

        # ── Inference ─────────────────────────────────────────────────────────
        history.append({"role": "user", "content": user_input})
        mcp_tools = [t for c in mcp_clients for t in c.get_tools()]
        mcp_executors = {k: v for c in mcp_clients for k, v in c.get_executors().items()}
        active_tools = mcp_tools or None
        active_executors = mcp_executors or None
        try:
            if mode == MODE_ASK:
                msgs = [{"role": "system", "content": system_prompt}, *history]
                answer = _run_ask(msgs, model_a, reasoning_effort=reasoning_effort_a)
            elif mode == MODE_PLAN:
                answer = _run_plan(
                    list(history),
                    model_a,
                    model_b,
                    reasoning_effort_a=reasoning_effort_a,
                    reasoning_effort_b=reasoning_effort_b,
                    tools=active_tools,
                    tool_executors=active_executors,
                )
            else:
                answer = _run_solve(
                    list(history),
                    model_a,
                    model_b,
                    reasoning_effort_a=reasoning_effort_a,
                    reasoning_effort_b=reasoning_effort_b,
                    tools=active_tools,
                    tool_executors=active_executors,
                )
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
