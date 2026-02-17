"""
inference.py — Low-level LiteLLM call and inference steps for HyperThink.
"""

import json
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

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
    temp_a_start: float
    temp_a_end: float
    temp_a_anneal_steps: Optional[int]
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
    _total_prompt_tokens: int
    _total_completion_tokens: int
    _total_cost_usd: float
    # Tool calling
    tools: Optional[List[Dict[str, Any]]]
    tool_registry: Dict[str, Callable[[Any], str]]
    max_tool_iterations: int

    def _log(self, msg: str) -> None: ...  # implemented in HyperThink

    # ------------------------------------------------------------------
    # Annealing
    # ------------------------------------------------------------------

    def _anneal_temp_a(self, step: int) -> float:
        """Return model A's annealed temperature at the given 0-indexed review step.

        Uses a linear schedule from ``temp_a_start`` down to ``temp_a_end``
        over ``temp_a_anneal_steps`` steps (default 10).  Beyond that the
        temperature is clamped to ``temp_a_end``.
        """
        T = self.temp_a_anneal_steps if self.temp_a_anneal_steps is not None else 10
        t = min(step, T)
        return self.temp_a_end + (self.temp_a_start - self.temp_a_end) * (1.0 - t / T)

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
        tools: Optional[List[Dict[str, Any]]] = None,
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
        if tools is not None:
            kwargs["tools"] = tools

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message="Pydantic serializer warnings"
            )
            response = litellm.completion(**kwargs)

        self._accumulate_usage(response)
        return response

    def _accumulate_usage(self, response: litellm.ModelResponse) -> None:
        """Extract token usage and cost from a response and add to running totals."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self._total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self._total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        try:
            self._total_cost_usd += litellm.completion_cost(completion_response=response)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def _dispatch_tool_call(self, tool_call) -> str:
        """Execute a single tool call and return the string result."""
        name: str = tool_call.function.name
        raw_args: str = tool_call.function.arguments or "{}"

        executor = self.tool_registry.get(name)
        if executor is None:
            return f"Error: unknown tool '{name}'."

        try:
            return executor(raw_args)
        except Exception as exc:
            return f"Error executing tool '{name}': {exc}"

    def _run_tool_loop(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        top_k: Optional[int],
        reasoning_effort: Optional[str],
        response_format: Optional[Dict] = None,
    ) -> litellm.ModelResponse:
        """
        Run an inference call followed by an agentic tool-call loop.

        If ``self.tools`` is empty / None the loop reduces to a single
        ``_call()`` invocation (identical behaviour to before).

        After all tool calls have been satisfied the final LLM response
        (which must contain plain text or JSON, never another tool call)
        is returned.  ``response_format`` is only applied to the **final**
        call so that intermediate tool-resolution calls are never forced
        into JSON mode.

        Parameters
        ----------
        model, messages, temperature, top_p, top_k, reasoning_effort:
            Forwarded to ``_call()``.
        response_format:
            Applied only on the final (non-tool) call.
        """
        active_tools = self.tools if self.tools else None

        # Local message list — we extend it with tool results without
        # mutating the caller's messages.
        local_messages = list(messages)

        for iteration in range(self.max_tool_iterations + 1):
            is_last_allowed = iteration >= self.max_tool_iterations

            # On the last allowed iteration drop tools to force a text reply.
            current_tools = None if is_last_allowed else active_tools
            # Apply response_format only when we're no longer passing tools
            # (some providers reject the combination).
            current_fmt = response_format if current_tools is None else None

            response = self._call(
                model=model,
                messages=local_messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
                response_format=current_fmt,
                tools=current_tools,
            )

            # Check for tool calls
            choice = response.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None)

            if not tool_calls:
                # No tool calls — this is the final text/JSON response.
                if current_fmt is None and response_format is not None:
                    # We deferred response_format; re-request with it now.
                    # Build the assistant turn without tool calls so the
                    # context is complete, then do a final structured call.
                    assistant_content = choice.message.content or ""
                    if assistant_content.strip():
                        local_messages.append(
                            {"role": "assistant", "content": assistant_content}
                        )
                    response = self._call(
                        model=model,
                        messages=local_messages,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        reasoning_effort=reasoning_effort,
                        response_format=response_format,
                        tools=None,
                    )
                return response

            if is_last_allowed:
                # Should not happen (tools=None forces text), but guard anyway.
                return response

            # Execute every tool call and collect results
            self._log(
                f"[HyperThink] Tool calls: "
                + ", ".join(tc.function.name for tc in tool_calls)
            )

            # Append the assistant message that contains the tool_calls
            local_messages.append(choice.message)

            for tc in tool_calls:
                result = self._dispatch_tool_call(tc)
                self._log(
                    f"[HyperThink] Tool '{tc.function.name}' → {result[:120]}"
                    + ("…" if len(result) > 120 else "")
                )
                local_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        # Unreachable, but satisfies type checkers
        return response  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Starter inference (Model A, first step)
    # ------------------------------------------------------------------

    def _run_starter(self, user_messages: List[Dict[str, Any]]) -> str:
        messages = [
            {"role": "system", "content": self.starter_prompt},
            *user_messages,
        ]
        self._log(f"[HyperThink] Starter inference → {self.model_a}")
        response = self._run_tool_loop(
            model=self.model_a,
            messages=messages,
            temperature=self._anneal_temp_a(0),
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
            response = self._run_tool_loop(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                reasoning_effort=reasoning_effort,
                response_format={"type": "json_object"},
            )
        except litellm.exceptions.BadRequestError as exc:
            self._log(
                f"[HyperThink] JSON response_format not supported by provider "
                f"({exc!r}), retrying without."
            )
            response = self._run_tool_loop(
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
