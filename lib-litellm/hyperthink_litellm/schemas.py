from dataclasses import dataclass

from pydantic import BaseModel, Field
from typing import List


@dataclass
class UsageStats:
    """Token usage and estimated cost for a single HyperThink query."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float

    def __str__(self) -> str:
        return (
            f"{self.total_tokens} tokens "
            f"(↑{self.prompt_tokens} prompt / ↓{self.completion_tokens} completion) · "
            f"${self.cost_usd:.6f}"
        )


class ReviewerOutput(BaseModel):
    review_result: bool = Field(
        ...,
        description="True if the answer is accepted as correct and complete.",
    )
    added_notes: List[str] = Field(
        default_factory=list,
        description="Notes to add to the auto-decaying state (2–8 when review_result is False, empty otherwise).",
    )
    output: str = Field(
        ...,
        description="The final or improved answer text.",
    )


class PlanOutput(BaseModel):
    tasks: List[str] = Field(
        ...,
        description="Ordered list of sub-tasks derived from the original query.",
    )
