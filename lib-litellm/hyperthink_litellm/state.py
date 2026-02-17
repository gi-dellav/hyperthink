import random
from typing import Callable, List, Optional


class AutoDecayingState:
    """Bounded list of notes that drops random entries when it would overflow."""

    def __init__(self, max_size: int = 17) -> None:
        assert max_size > 0, "max_size must be a positive integer"
        self.max_size: int = max_size
        self.notes: List[str] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_notes(
        self,
        new_notes: List[str],
        log: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Append *new_notes*, evicting random existing notes if necessary."""
        assert isinstance(new_notes, list), "new_notes must be a list"
        assert all(isinstance(n, str) for n in new_notes), "every note must be a string"

        # Truncate incoming batch if it alone exceeds the capacity.
        if len(new_notes) > self.max_size:
            new_notes = new_notes[-self.max_size:]

        available = self.max_size - len(self.notes)
        overflow = len(new_notes) - available

        if overflow > 0:
            evict_indices = sorted(
                random.sample(range(len(self.notes)), overflow), reverse=True
            )
            evicted = [self.notes[i] for i in evict_indices]
            for i in evict_indices:
                del self.notes[i]
            if log:
                log(f"[State] Evicted {overflow} random note(s): {evicted}")

        self.notes.extend(new_notes)

        if log:
            log(
                f"[State] Added {len(new_notes)} note(s). "
                f"State: {len(self.notes)}/{self.max_size}."
            )

    def clear(self) -> None:
        self.notes.clear()

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format(self) -> str:
        """Return notes as a numbered list, or '(none)' when empty."""
        if not self.notes:
            return "(none)"
        return "\n".join(f"{i + 1}. {note}" for i, note in enumerate(self.notes))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"max_size": self.max_size, "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, data: dict) -> "AutoDecayingState":
        assert (
            "max_size" in data and "notes" in data
        ), "Checkpoint data must contain 'max_size' and 'notes' keys"
        obj = cls(max_size=data["max_size"])
        obj.notes = list(data["notes"])
        return obj

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.notes)

    def __repr__(self) -> str:
        return f"AutoDecayingState(max_size={self.max_size}, notes={len(self.notes)})"
