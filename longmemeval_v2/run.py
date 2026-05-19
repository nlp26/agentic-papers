"""Run the 15 questions through both modes (no_memory vs with_memory), score
with the judge per category, print a side-by-side table.

Mock provider differentiates: with_memory mode synthesizes from retrieved
facts; no_memory mode returns 'I don't know'. With a real LLM both modes
produce reasoned answers and the gap reflects the model's prior knowledge of
the synthetic environment (which is none — that's the point).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import get_provider, score_task  # noqa: E402
from agent import answer_no_memory, answer_with_memory  # noqa: E402
from memory import ExperienceMemory  # noqa: E402
from mock_provider import MockMemoryProvider  # noqa: E402
from rubric import OPERATOR_ANSWER  # noqa: E402


CATEGORY_TO_HINT = {
    "static_state": None,
    "dynamic_state": "dynamic",
    "workflow": "workflow",
    "gotcha": "gotcha",
    "premise": "premise",
}


def _agent_provider():
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return get_provider()
    return MockMemoryProvider()


def _score(judge, q: dict, answer: str) -> float:
    return score_task(
        provider=judge,
        task=q["question"],
        candidate=answer,
        gold=q["gold"],
        rubric=OPERATOR_ANSWER,
        task_id=q["id"],
    ).aggregate()


def main() -> int:
    agent = _agent_provider()
    judge = get_provider()
    print(f"agent provider: {agent.name}")
    print(f"judge provider: {judge.name}\n")

    mem = ExperienceMemory.from_trajectories(HERE / "trajectories.json")
    qs = json.loads((HERE / "questions.json").read_text())["questions"]

    rows = []
    per_cat: dict[str, dict[str, list[float]]] = {}
    for q in qs:
        hint = CATEGORY_TO_HINT.get(q["category"])
        a_no = answer_no_memory(agent, q["question"])
        a_yes = answer_with_memory(agent, mem, q["question"], hint_tag=hint)
        s_no = _score(judge, q, a_no)
        s_yes = _score(judge, q, a_yes)
        rows.append({
            "id": q["id"], "cat": q["category"],
            "no_mem": s_no, "with_mem": s_yes,
        })
        per_cat.setdefault(q["category"], {"no": [], "yes": []})
        per_cat[q["category"]]["no"].append(s_no)
        per_cat[q["category"]]["yes"].append(s_yes)

    print(f"{'id':<14} {'category':<14} {'no_mem':>8} {'with_mem':>10} {'Δ':>7}")
    print("-" * 60)
    for r in rows:
        d = r["with_mem"] - r["no_mem"]
        print(f"{r['id']:<14} {r['cat']:<14} {r['no_mem']:>8.2f} {r['with_mem']:>10.2f} {d:>+7.2f}")

    print()
    print(f"{'BY CATEGORY':<14} {'':<14} {'no_mem':>8} {'with_mem':>10} {'Δ':>7}")
    print("-" * 60)
    for cat in ["static_state", "dynamic_state", "workflow", "gotcha", "premise"]:
        if cat not in per_cat:
            continue
        n = per_cat[cat]["no"]
        y = per_cat[cat]["yes"]
        avg_n = sum(n) / len(n)
        avg_y = sum(y) / len(y)
        print(f"{'':<14} {cat:<14} {avg_n:>8.2f} {avg_y:>10.2f} {avg_y - avg_n:>+7.2f}")

    all_n = [s for c in per_cat.values() for s in c["no"]]
    all_y = [s for c in per_cat.values() for s in c["yes"]]
    print("-" * 60)
    print(f"{'OVERALL':<14} {'':<14} {sum(all_n)/len(all_n):>8.2f} {sum(all_y)/len(all_y):>10.2f} "
          f"{(sum(all_y)/len(all_y)) - (sum(all_n)/len(all_n)):>+7.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
