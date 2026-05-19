"""Rubrics. Add new ones here per task type."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Criterion, Rubric  # noqa: E402


QA_SHORT_ANSWER = Rubric(
    name="qa_short_answer",
    criteria=[
        Criterion(
            name="faithfulness",
            description="Only states facts supported by the gold or task context.",
            anchors={
                0: "Contradicts gold or invents facts.",
                3: "Mostly faithful, one minor unsupported claim.",
                5: "Every claim directly supported.",
            },
        ),
        Criterion(
            name="completeness",
            description="Covers the key points the task asks for.",
            anchors={
                0: "Misses the central point.",
                3: "Central point present, important detail missing.",
                5: "Central point and supporting detail present.",
            },
        ),
        Criterion(
            name="reasoning",
            description="Reasoning is sound, ordered, and traceable.",
            anchors={
                0: "No reasoning, or incoherent.",
                3: "Present but skips a step.",
                5: "Crisp and traceable.",
            },
        ),
    ],
    scale=(0, 5),
)
