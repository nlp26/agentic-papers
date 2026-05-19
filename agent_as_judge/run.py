"""Run the judge over 10 QA tasks (strong + weak candidate each), print scores,
measure agreement vs human labels, derive bias, calibrate, measure again.

Provider: auto from ANTHROPIC_API_KEY / OPENAI_API_KEY env, else mock.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import (
    agreement,
    apply_calibration,
    calibrate,
    get_provider,
    score_batch,
)
from rubrics import QA_SHORT_ANSWER


def _load():
    data = json.loads((HERE / "sample_tasks.json").read_text())
    flat = []
    for t in data["tasks"]:
        for variant, answer in t["candidates"].items():
            flat.append({
                "id": f"{t['id']}_{variant}",
                "task": t["task"],
                "gold": t["gold"],
                "candidate": answer,
            })
    return flat, data["human_labels"]


def _fmt(jr):
    parts = " ".join(f"{c}={s:.1f}" for c, s in jr.scores.items())
    return f"{parts}   -> {jr.aggregate():.2f}"


def main(provider_name: str | None = None) -> int:
    provider = get_provider(provider_name)
    print(f"provider: {provider.name}")
    print(f"rubric:   {QA_SHORT_ANSWER.name}\n")

    tasks, human = _load()
    results = score_batch(provider=provider, tasks=tasks, rubric=QA_SHORT_ANSWER)

    print("scores (uncalibrated):")
    for jr in results:
        print(f"  {jr.task_id:<20} {_fmt(jr)}")
    print()

    agr = agreement(results, human, tolerance=0.5)
    print("agreement vs human (tol=0.5):")
    for k, v in agr.items():
        print(f"  {k:<14} {v:.2f}")
    print()

    bias = calibrate(results, human)
    print("per-criterion bias (judge - human):")
    for k, v in bias.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k:<14} {sign}{v:.2f}")
    print()

    agr2 = agreement(apply_calibration(results, bias), human, tolerance=0.5)
    print("agreement after calibration:")
    for k, v in agr2.items():
        print(f"  {k:<14} {v:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(os.getenv("JUDGE_PROVIDER")))
