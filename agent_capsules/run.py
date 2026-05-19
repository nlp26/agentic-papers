"""Run the same scenarios through baseline and capsule modes, judge both outputs
against the gold summary, print a side-by-side token + quality table.

With the mock provider only the token side is meaningful (mock returns the same
text regardless of input). With a real provider both sides are.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import get_provider, score_task  # noqa: E402
from capsule import run_pipeline  # noqa: E402
from mock_pipeline import MockPipelineProvider  # noqa: E402
from rubric import DUE_DILIGENCE  # noqa: E402


def _pipeline_provider():
    # Same auto-detect as the judge, but mock here is the pipeline-aware one.
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return get_provider()
    return MockPipelineProvider()


def _load_scenarios() -> list[dict]:
    data = json.loads((HERE / "scenarios.json").read_text())
    return data["scenarios"]


def _score(judge, scenario: dict, summary: str) -> float:
    jr = score_task(
        provider=judge,
        task=scenario["question"],
        candidate=summary,
        gold=scenario["gold_summary"],
        rubric=DUE_DILIGENCE,
        task_id=scenario["id"],
    )
    return jr.aggregate()


def main() -> int:
    pipeline = _pipeline_provider()
    judge = get_provider()

    print(f"pipeline provider: {pipeline.name}")
    print(f"judge provider:    {judge.name}")
    print()

    scenarios = _load_scenarios()

    rows = []
    for sc in scenarios:
        base = run_pipeline(pipeline, sc, mode="baseline")
        caps = run_pipeline(pipeline, sc, mode="capsule")
        q_base = _score(judge, sc, base["summary"])
        q_caps = _score(judge, sc, caps["summary"])
        rows.append({
            "id": sc["id"],
            "tokens_base": base["tokens"],
            "tokens_caps": caps["tokens"],
            "escalations": caps["escalations"],
            "quality_base": q_base,
            "quality_caps": q_caps,
        })

    print(f"{'scenario':<20} {'base_tok':>9} {'caps_tok':>9} {'Δtok%':>7} "
          f"{'esc':>4} {'q_base':>7} {'q_caps':>7} {'Δq':>6}")
    print("-" * 80)

    sum_b = sum_c = 0
    qb_tot = qc_tot = 0.0
    for r in rows:
        d_tok = (r["tokens_caps"] - r["tokens_base"]) / r["tokens_base"] * 100
        d_q = r["quality_caps"] - r["quality_base"]
        print(f"{r['id']:<20} {r['tokens_base']:>9} {r['tokens_caps']:>9} "
              f"{d_tok:>+6.1f}% {r['escalations']:>4} "
              f"{r['quality_base']:>7.2f} {r['quality_caps']:>7.2f} {d_q:>+6.2f}")
        sum_b += r["tokens_base"]
        sum_c += r["tokens_caps"]
        qb_tot += r["quality_base"]
        qc_tot += r["quality_caps"]

    n = len(rows)
    d_tot = (sum_c - sum_b) / sum_b * 100
    print("-" * 80)
    print(f"{'TOTAL':<20} {sum_b:>9} {sum_c:>9} {d_tot:>+6.1f}% "
          f"{'':>4} {qb_tot / n:>7.2f} {qc_tot / n:>7.2f} {qc_tot / n - qb_tot / n:>+6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
