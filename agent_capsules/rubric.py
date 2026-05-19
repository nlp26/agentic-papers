"""Rubric for due-diligence summary quality."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Criterion, Rubric  # noqa: E402


DUE_DILIGENCE = Rubric(
    name="due_diligence_summary",
    criteria=[
        Criterion(
            name="factual_grounding",
            description="Claims in the summary are supported by the source/gold facts.",
            anchors={
                0: "Invents or contradicts source facts.",
                3: "Mostly grounded, one minor unsupported claim.",
                5: "Every claim is grounded.",
            },
        ),
        Criterion(
            name="coverage",
            description="Touches the main sub-topics the question asked about.",
            anchors={
                0: "Misses entire sub-topics.",
                3: "Sub-topics present but one is shallow.",
                5: "All sub-topics covered with the strongest item from each.",
            },
        ),
        Criterion(
            name="risk_opportunity",
            description="Names a specific strongest risk and strongest opportunity, not generic ones.",
            anchors={
                0: "No risk or opportunity named, or both are generic.",
                3: "One is specific, the other generic.",
                5: "Both are specific and traceable to the source.",
            },
        ),
    ],
    scale=(0, 5),
)
