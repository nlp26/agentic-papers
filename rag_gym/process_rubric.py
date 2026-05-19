"""Process-supervision scoring. Score each intermediate step independently."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import Criterion, Rubric  # noqa: E402


QUERY_QUALITY = Rubric(
    name="query_quality",
    criteria=[
        Criterion(
            name="specificity",
            description="The rewritten query is more specific than the raw question and removes filler.",
            anchors={
                0: "Identical to the question, includes stopwords.",
                3: "Slightly tightened.",
                5: "Sharp keyword set, no stopwords, targets the answer.",
            },
        ),
    ],
)


RETRIEVAL_RELEVANCE = Rubric(
    name="retrieval_relevance",
    criteria=[
        Criterion(
            name="contains_gold_doc",
            description="The retrieved set contains the gold document id.",
            anchors={
                0: "Gold doc not in top-k.",
                3: "Gold doc in top-k but not at rank 1.",
                5: "Gold doc at rank 1.",
            },
        ),
    ],
)


REASONING_GROUNDING = Rubric(
    name="reasoning_grounding",
    criteria=[
        Criterion(
            name="cites_correctly",
            description="The answer cites the gold document and the answer text is supported by it.",
            anchors={
                0: "No citation, or cites the wrong doc, or contradicts the gold.",
                3: "Cites the gold doc but answer is partially supported.",
                5: "Cites the gold doc and the answer matches the gold answer.",
            },
        ),
    ],
)


REFLECTION_CALIBRATION = Rubric(
    name="reflection_calibration",
    criteria=[
        Criterion(
            name="verdict_matches_truth",
            description="The reflect verdict matches whether the answer is actually correct.",
            anchors={
                0: "Verdict opposite of truth (false confidence or false doubt).",
                3: "Verdict ambiguous.",
                5: "Verdict matches truth.",
            },
        ),
    ],
)


FINAL_ANSWER = Rubric(
    name="final_answer",
    criteria=[
        Criterion(
            name="matches_gold",
            description="The final answer conveys the same facts as the gold answer.",
            anchors={
                0: "Wrong or unrelated.",
                3: "Right direction, missing key detail.",
                5: "Matches the gold answer.",
            },
        ),
    ],
)
