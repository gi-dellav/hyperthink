import litellm
from rich.markdown import Markdown

from hyperthink_litellm import HyperThink, UsageStats

from .constants import console


class _RichHyperThink(HyperThink):
    """HyperThink that routes _log() through the rich console."""

    def _log(self, msg: str) -> None:
        if self.logging_enabled:
            console.log(f"[dim]{msg}[/dim]")


def _run_ask(
    messages: list,
    model: str,
    reasoning_effort: str | None = None,
) -> str:
    """Stream a direct LiteLLM inference; return the full response text."""
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    response = litellm.completion(**kwargs)
    full_text = ""
    last_usage = None
    console.print()
    for chunk in response:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_text += delta
            print(delta, end="", flush=True)
        usage = getattr(chunk, "usage", None)
        if usage:
            last_usage = usage
    print()
    console.print()
    if last_usage:
        try:
            cost = litellm.completion_cost(
                model=model,
                prompt_tokens=getattr(last_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(last_usage, "completion_tokens", 0) or 0,
            )
            stats = UsageStats(
                prompt_tokens=getattr(last_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(last_usage, "completion_tokens", 0) or 0,
                total_tokens=(getattr(last_usage, "prompt_tokens", 0) or 0)
                + (getattr(last_usage, "completion_tokens", 0) or 0),
                cost_usd=cost,
            )
            console.print(f"[dim]{stats}[/dim]")
            console.print()
        except Exception:
            pass
    return full_text


def _run_solve(
    messages: list,
    model_a: str,
    model_b: str,
    reasoning_effort_a: str | None = None,
    reasoning_effort_b: str | None = None,
) -> str:
    """Run a HyperThink scaffolding query; return the final answer."""
    ht = _RichHyperThink(
        model_a=model_a,
        model_b=model_b,
        reasoning_effort_a=reasoning_effort_a,
        reasoning_effort_b=reasoning_effort_b,
        logging_enabled=True,
    )
    console.print()
    result = ht.query(messages)
    console.print()
    console.print(Markdown(result))
    console.print()
    console.print(f"[dim]{ht.last_usage}[/dim]")
    console.print()
    return result
