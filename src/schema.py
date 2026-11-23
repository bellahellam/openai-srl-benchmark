"""Shared data schema for PropBank-style SRL.

One raw dataset example describes SRL for a single predicate in a sentence, so
one SRLExample = one sentence + one predicate + its arguments.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class Argument(BaseModel):
    """A single semantic argument of a predicate."""

    text: str = Field(description="Exact surface text of the argument span.")
    role: str = Field(description="PropBank role label, e.g. ARG0, ARG1, ARGM-TMP.")
    start: Optional[int] = Field(default=None)
    end: Optional[int] = Field(default=None)


class Prediction(BaseModel):
    """What the model returns for one (sentence, predicate) pair."""

    predicate: str
    arguments: List[Argument] = Field(default_factory=list)


class SRLExample(BaseModel):
    """A fully specified gold example used throughout the pipeline."""

    example_id: str
    sentence: str
    predicate: str
    predicate_start: int
    predicate_end: int
    arguments: List[Argument] = Field(default_factory=list)

    def gold_tuples(self) -> set:
        """Exact labeled-span tuples: (id, p_start, p_end, a_start, a_end, role)."""
        return {
            (
                self.example_id,
                self.predicate_start,
                self.predicate_end,
                a.start,
                a.end,
                a.role,
            )
            for a in self.arguments
            if a.start is not None and a.end is not None
        }
