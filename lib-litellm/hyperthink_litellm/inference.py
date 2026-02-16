"""
inference.py — Low-level LiteLLM call and inference steps for HyperThink.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import litellm

from .helpers import _extract_json, _format_reviewer_prompt
from .schemas import ReviewerOutput

if TYPE_CHECKING:
    from .state import AutoDecayingState


class _InferenceMixin:
    """Mixin providing low-level inference methods for HyperThink."""

    # These attributes are set by HyperThink.__init__; declared here for type checkers.
    model_a: str
    model_b: str
    temp_a: float
    temp_b: float
    top_p_a: float
    top_p_b: float
    top_k_a: Optional[int]
    top_k_b: Optional[int]
    starter_prompt: str
    reviewer_prompt: str
    reasoning_effort_a: Optional[str]
    reasoning_effort_b: Optional[str]
    state: "AutoDecayingState"
    logging_enabled: bool

    def _log(self, msg: str) -> None: ...  # implemented in HyperThink

    # ------------------------------------------------------------------
    # Low-level inference
    # ------------------------------------------------------------------

    def _call(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        reasoning_effort: Optional[str],
        response_format: Optional[Dict] = None,
    ) -> litellm.ModelResponse:
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if top_k is not None:
            kwargs["top_k"] = top_k
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            kwargs["response_format"] = response_format

        return litellm.completion(**kwargs)

    # ------------------------------------------------------------------
    # Starter inference (Model A, first step)
    # ------------------------------------------------------------------

    def _run_starter(self, user_messages: List[Dict[str, Any]]) -> str:
        messages = [
            {"role": "system", "content": self.starter_prompt},
            *user_messages,
        ]
        self._log(f"[HyperThink] Starter inference → {self.model_a}")
        response = self._call(
            model=self.model_a,
            messages=messages,
            temperature=self.temp_a,
            top_p=self.top_p_a,
            top_k=self.top_k_a,
            reasoning_effort=self.reasoning_effort_a,
        )
        content = response.choices[0].message.content
        assert (
            content is not None and content.strip()
        ), "Starter model returned empty content"
        return content

    # ------------------------------------------------------------------
    # Reviewer inference
    # ------------------------------------------------------------------

    def _run_reviewer(
        self,
        model: str,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        reasoning_effort: Optional[str],
        user_messages: List[Dict[str, Any]],
        current_answer: str,
    ) -> ReviewerOutput:
        system_prompt = _format_reviewer_prompt(
            self.reviewer_prompt,
            notes=self.state.format(),
            review_input=current_answer,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *user_messages,
        ]

        # Request JSON output; fall back gracefully if the provider rejects it.
        try:
            response = self._call(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = self._call(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
            )

        content = response.choices[0].message.content
        assert (
            content is not None and content.strip()
        ), "Reviewer model returned empty content"

        try:
            data = json.loads(_extract_json(content))
            result = ReviewerOutput(**data)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse reviewer output as ReviewerOutput.\n"
                f"Error: {exc}\nRaw content:\n{content}"
            ) from exc

        assert isinstance(result.review_result, bool), "review_result must be a boolean"
        if not result.review_result:
            assert 2 <= len(result.added_notes) <= 8, (
                f"added_notes must contain 2–8 items when review_result is False "
                f"(got {len(result.added_notes)})"
            )

        return result
