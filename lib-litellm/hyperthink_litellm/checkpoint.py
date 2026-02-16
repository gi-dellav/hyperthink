"""
checkpoint.py — Checkpoint persistence and state-reset mixin for HyperThink.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .state import AutoDecayingState


class _CheckpointMixin:
    """Mixin providing checkpoint save/load and state reset for HyperThink."""

    # Declared here for type checkers; set by HyperThink.__init__.
    max_state_size: int
    state: "AutoDecayingState"
    iteration_count: int

    def _log(self, msg: str) -> None: ...  # implemented in HyperThink

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Persist the current scaffolding state to a JSON file."""
        checkpoint = {
            "state": self.state.to_dict(),
            "iteration_count": self.iteration_count,
            "config": {
                "model_a": self.model_a,
                "model_b": self.model_b,
                "max_state_size": self.max_state_size,
                "max_iterations": self.max_iterations,
                "temp_a": self.temp_a,
                "temp_b": self.temp_b,
                "top_p_a": self.top_p_a,
                "top_p_b": self.top_p_b,
                "top_k_a": self.top_k_a,
                "top_k_b": self.top_k_b,
                "reasoning_effort_a": self.reasoning_effort_a,
                "reasoning_effort_b": self.reasoning_effort_b,
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, indent=2)
        self._log(f"[HyperThink] Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Restore scaffolding state from a JSON checkpoint file."""
        from .state import AutoDecayingState

        with open(path, encoding="utf-8") as fh:
            checkpoint = json.load(fh)
        assert (
            "state" in checkpoint and "iteration_count" in checkpoint
        ), "Checkpoint file is missing required keys ('state', 'iteration_count')"
        self.state = AutoDecayingState.from_dict(checkpoint["state"])
        self.iteration_count = checkpoint["iteration_count"]
        self._log(f"[HyperThink] Checkpoint loaded ← {path}")

    def reset(self) -> None:
        """Clear runtime state (notes + iteration counter)."""
        from .state import AutoDecayingState

        self.state = AutoDecayingState(max_size=self.max_state_size)
        self.iteration_count = 0
        self._log("[HyperThink] State reset.")
