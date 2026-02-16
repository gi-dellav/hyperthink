"""
hyperthink-litellm
==================
HyperThink LLM scaffolding algorithm implemented on top of LiteLLM.

Quick start
-----------
    from hyperthink import query, HyperThink

    # One-shot convenience function
    answer = query([{"role": "user", "content": "Explain Gödel's incompleteness theorems."}])

    # Full control via the class
    ht = HyperThink(model_a="deepseek/deepseek-chat", logging_enabled=True)
    answer = ht.query([{"role": "user", "content": "What is 17 * 23?"}])
"""

from .defaults import DEFAULT_MODEL_A, DEFAULT_MODEL_B
from .hyperthink import HyperThink
from .prompts import REVIEWER_PROMPT, STARTER_PROMPT
from .schemas import ReviewerOutput
from .state import AutoDecayingState


def query(
    messages: list,
    *,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    max_state_size: int = 17,
    max_iterations: int | None = None,
    temp_a: float = 1.2,
    temp_b: float = 0.0,
    top_p_a: float = 0.95,
    top_p_b: float = 0.2,
    top_k_a: int | None = None,
    top_k_b: int | None = None,
    starter_prompt: str = STARTER_PROMPT,
    reviewer_prompt: str = REVIEWER_PROMPT,
    reasoning_effort_a: str | None = None,
    reasoning_effort_b: str | None = None,
    logging_enabled: bool = False,
) -> str:
    """
    Execute a query using the HyperThink scaffolding.

    This is a stateless convenience wrapper around :class:`HyperThink`.
    For repeated queries or checkpoint support, instantiate the class directly.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-style message list (``[{"role": "user", "content": "..."}]``).
    model_a : str
        Starter / odd-review model (default: Deepseek V3).
    model_b : str
        Even-review model (default: Gemini Flash).
    max_state_size : int
        Maximum notes in the auto-decaying state.
    max_iterations : int | None
        Hard cap on total inference calls; ``None`` = unlimited.
    temp_a, temp_b : float
        Sampling temperatures.
    top_p_a, top_p_b : float
        Nucleus-sampling thresholds.
    top_k_a, top_k_b : int | None
        Top-k values (forwarded when not ``None``).
    starter_prompt : str
        System prompt for the first inference.
    reviewer_prompt : str
        System prompt template for review inferences.
    reasoning_effort_a, reasoning_effort_b : str | None
        Reasoning effort hints forwarded to LiteLLM.
    logging_enabled : bool
        Print progress to stdout.

    Returns
    -------
    str
        The final reviewed answer.
    """
    ht = HyperThink(
        model_a=model_a,
        model_b=model_b,
        max_state_size=max_state_size,
        max_iterations=max_iterations,
        temp_a=temp_a,
        temp_b=temp_b,
        top_p_a=top_p_a,
        top_p_b=top_p_b,
        top_k_a=top_k_a,
        top_k_b=top_k_b,
        starter_prompt=starter_prompt,
        reviewer_prompt=reviewer_prompt,
        reasoning_effort_a=reasoning_effort_a,
        reasoning_effort_b=reasoning_effort_b,
        logging_enabled=logging_enabled,
    )
    return ht.query(messages)


__all__ = [
    "HyperThink",
    "query",
    "AutoDecayingState",
    "ReviewerOutput",
    "STARTER_PROMPT",
    "REVIEWER_PROMPT",
    "DEFAULT_MODEL_A",
    "DEFAULT_MODEL_B",
]
