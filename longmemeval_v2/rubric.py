"""Rubric for operator-style answers grounded in remembered facts."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Criterion, Rubric  # noqa: E402


OPERATOR_ANSWER = Rubric(
    name="operator_answer",
    criteria=[
        Criterion(
            name="factual_match",
            description="Does the answer state the same facts as the gold answer for this category?",
            anchors={
                0: "Wrong, unrelated, or 'I don't know' when the gold is concrete.",
                3: "Right direction, missing key detail.",
                5: "Matches the gold answer's content.",
            },
        ),
        Criterion(
            name="actionable",
            description="Could an on-call operator act on this answer (correct command, correct ordering, correct constraint)?",
            anchors={
                0: "Generic or non-actionable.",
                3: "Implies the right action but lacks detail.",
                5: "Names commands, ordering, or constraints precisely.",
            },
        ),
    ],
    scale=(0, 5),
)
