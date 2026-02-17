"""
hyperthink.py — HyperThink scaffolding engine backed by LiteLLM.

Algorithm recap
---------------
1. Model A (high-temp, high top_p) generates the initial answer.
2. Model B (low-temp, low top_p) reviews with structured output.
   • If accepted → return output.
   • If rejected → update state with added_notes, pass B's improved output to…
3. Model A reviews (same structured format as B).
   • If accepted → return output.
   • If rejected → update state, pass A's output back to step 2.
4. Steps 2–3 alternate until accepted or the iteration limit is reached.
"""

from typing import Any, Dict, List, Optional

from .checkpoint import _CheckpointMixin
from .defaults import DEFAULT_MODEL_A, DEFAULT_MODEL_B
from .inference import _InferenceMixin
from .prompts import REVIEWER_PROMPT, STARTER_PROMPT
from .schemas import UsageStats
from .state import AutoDecayingState


class HyperThink(_InferenceMixin, _CheckpointMixin):
    """
    HyperThink scaffolding manager.

    Parameters
    ----------
    model_a : str
        Model used for the initial (starter) inference and odd-numbered reviews.
    model_b : str
        Model used for even-numbered reviews.
    max_state_size : int
        Maximum number of notes in the auto-decaying state (default 17).
    max_iterations : int | None
        Hard cap on total inference calls.  When reached the current answer is
        returned as-is.  ``None`` means unlimited.
    temp_a_start : float
        Starting temperature for model A (annealing schedule begins here).
    temp_a_end : float
        Final temperature model A anneals down to.
    temp_a_anneal_steps : int | None
        Number of review steps over which model A's temperature decays from
        ``temp_a_start`` to ``temp_a_end``.  Uses a linear schedule.
        Defaults to ``max_iterations`` when set, otherwise 10.
    temp_b : float
        Fixed sampling temperature for model B.
    top_p_a, top_p_b : float
        Top-p (nucleus sampling) values for model A and B.
    top_k_a, top_k_b : int | None
        Top-k values passed as extra LiteLLM kwargs when set.
    starter_prompt : str
        System prompt used for the first inference.
    reviewer_prompt : str
        System prompt template used for all review inferences.
        Must contain ``{notes}`` and ``{review_input}`` placeholders.
    reasoning_effort_a, reasoning_effort_b : str | None
        Reasoning effort hint forwarded to LiteLLM (e.g. ``"high"``).
    logging_enabled : bool
        When ``True`` progress is printed to stdout.
    """

    def __init__(
        self,
        model_a: str = DEFAULT_MODEL_A,
        model_b: str = DEFAULT_MODEL_B,
        max_state_size: int = 17,
        max_iterations: Optional[int] = None,
        temp_a_start: float = 1.6,
        temp_a_end: float = 0.2,
        temp_a_anneal_steps: Optional[int] = None,
        temp_b: float = 0.0,
        top_p_a: float = 0.95,
        top_p_b: float = 0.2,
        top_k_a: Optional[int] = None,
        top_k_b: Optional[int] = None,
        starter_prompt: str = STARTER_PROMPT,
        reviewer_prompt: str = REVIEWER_PROMPT,
        reasoning_effort_a: Optional[str] = None,
        reasoning_effort_b: Optional[str] = None,
        logging_enabled: bool = False,
    ) -> None:
        assert max_state_size > 0, "max_state_size must be a positive integer"
        assert (
            max_iterations is None or max_iterations > 0
        ), "max_iterations must be a positive integer or None"
        assert 0.0 <= temp_a_start, "temp_a_start must be non-negative"
        assert 0.0 <= temp_a_end, "temp_a_end must be non-negative"
        assert temp_a_end <= temp_a_start, "temp_a_end must be <= temp_a_start"
        assert (
            temp_a_anneal_steps is None or temp_a_anneal_steps > 0
        ), "temp_a_anneal_steps must be a positive integer or None"
        assert 0.0 <= temp_b, "temp_b must be non-negative"
        assert 0.0 < top_p_a <= 1.0, "top_p_a must be in (0, 1]"
        assert 0.0 < top_p_b <= 1.0, "top_p_b must be in (0, 1]"
        assert (
            "{notes}" in reviewer_prompt
        ), "reviewer_prompt must contain the {notes} placeholder"
        assert (
            "{review_input}" in reviewer_prompt
        ), "reviewer_prompt must contain the {review_input} placeholder"

        self.model_a = model_a
        self.model_b = model_b
        self.max_state_size = max_state_size
        self.max_iterations = max_iterations
        self.temp_a_start = temp_a_start
        self.temp_a_end = temp_a_end
        self.temp_a_anneal_steps = (
            temp_a_anneal_steps if temp_a_anneal_steps is not None else max_iterations
        )
        self.temp_b = temp_b
        self.top_p_a = top_p_a
        self.top_p_b = top_p_b
        self.top_k_a = top_k_a
        self.top_k_b = top_k_b
        self.starter_prompt = starter_prompt
        self.reviewer_prompt = reviewer_prompt
        self.reasoning_effort_a = reasoning_effort_a
        self.reasoning_effort_b = reasoning_effort_b
        self.logging_enabled = logging_enabled

        # Runtime state — reset at the beginning of every query()
        self.state: AutoDecayingState = AutoDecayingState(max_size=max_state_size)
        self.iteration_count: int = 0

        # Cost / usage tracking — reset at the beginning of every query()
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.logging_enabled:
            print(msg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_usage(self) -> UsageStats:
        """Token usage and estimated cost accumulated during the last :meth:`query` call."""
        return UsageStats(
            prompt_tokens=self._total_prompt_tokens,
            completion_tokens=self._total_completion_tokens,
            total_tokens=self._total_prompt_tokens + self._total_completion_tokens,
            cost_usd=self._total_cost_usd,
        )

    def query(self, messages: List[Dict[str, Any]]) -> str:
        """
        Execute a query using the HyperThink scaffolding.

        Parameters
        ----------
        messages : list[dict]
            The real user conversation (same format as LiteLLM / OpenAI messages).
            Only these messages are kept as chat history; all intermediate
            scaffolding steps are invisible to the models as history.

        Returns
        -------
        str
            The final, reviewed answer.
        """
        assert (
            isinstance(messages, list) and len(messages) > 0
        ), "messages must be a non-empty list"

        # Fresh state for every query
        self.state = AutoDecayingState(max_size=self.max_state_size)
        self.iteration_count = 0
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_cost_usd = 0.0

        self._log("[HyperThink] ── Starting query ──────────────────────────────")

        # Step 1: starter inference with Model A
        current_answer = self._run_starter(messages)
        self.iteration_count += 1
        self._log(
            f"[HyperThink] Starter done. Answer length: {len(current_answer)} chars."
        )

        # Reviewer cycle: B, A, B, A, …
        # reviewer_cycle[0] = Model B params, reviewer_cycle[1] = Model A params
        # Model A's temperature is omitted here — it is computed per-step via annealing.
        reviewer_cycle = [
            (self.model_b, self.top_p_b, self.top_k_b, self.reasoning_effort_b, "B"),
            (self.model_a, self.top_p_a, self.top_k_a, self.reasoning_effort_a, "A"),
        ]
        review_step = 0   # cycles through 0, 1, 0, 1, …
        a_review_count = 0  # counts model-A review calls, used for annealing schedule

        while True:
            if (
                self.max_iterations is not None
                and self.iteration_count >= self.max_iterations
            ):
                self._log(
                    f"[HyperThink] Iteration limit ({self.max_iterations}) reached. "
                    "Returning current answer."
                )
                self._log(f"[HyperThink] Usage: {self.last_usage}")
                return current_answer

            model, top_p, top_k, reasoning_effort, label = reviewer_cycle[
                review_step % 2
            ]
            temp = self._anneal_temp_a(a_review_count) if label == "A" else self.temp_b
            if label == "A":
                self._log(f"[HyperThink] Model A temperature (annealed): {temp:.4f}")
            review_step += 1

            self._log(f"[HyperThink] Review #{review_step} → Model {label} ({model})")
            result = self._run_reviewer(
                model=model,
                temperature=temp,
                top_p=top_p,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
                user_messages=messages,
                current_answer=current_answer,
            )
            self.iteration_count += 1
            if label == "A":
                a_review_count += 1

            if result.review_result:
                self._log(
                    f"[HyperThink] ✓ Accepted after {self.iteration_count} inference(s)."
                )
                self._log(f"[HyperThink] Usage: {self.last_usage}")
                return result.output

            self._log(
                f"[HyperThink] ✗ Rejected. Adding {len(result.added_notes)} note(s)."
            )
            self.state.add_notes(
                result.added_notes,
                log=self._log if self.logging_enabled else None,
            )
            current_answer = result.output
