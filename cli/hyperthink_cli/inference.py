import litellm
from rich.markdown import Markdown

from hyperthink_litellm import HyperThink

from .constants import console


class _RichHyperThink(HyperThink):
    """HyperThink that routes _log() through the rich console."""

    def _log(self, msg: str) -> None:
        if self.logging_enabled:
            console.log(f"[dim]{msg}[/dim]")


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
