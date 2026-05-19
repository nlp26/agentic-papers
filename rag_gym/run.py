"""Run naive RAG and Re²Search over the question set, score each step with the
judge using process-supervision rubrics, print a per-step table.

Process supervision: a step that fails in isolation is visible here even when
the final answer happens to be right (or the final score happens to look fine).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _common import get_provider, score_task  # noqa: E402
from mock_rag import MockRagProvider  # noqa: E402
from process_rubric import (  # noqa: E402
    FINAL_ANSWER,
    QUERY_QUALITY,
    REASONING_GROUNDING,
    REFLECTION_CALIBRATION,
    RETRIEVAL_RELEVANCE,
)
from re2search import run_naive, run_re2search  # noqa: E402
from retriever import BM25, load_corpus  # noqa: E402


def _pipeline_provider():
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return get_provider()
    return MockRagProvider()


def _score(judge, *, task: str, candidate: str, gold: str, rubric, task_id: str) -> float:
    return score_task(
        provider=judge,
        task=task,
        candidate=candidate,
        gold=gold,
        rubric=rubric,
        task_id=task_id,
    ).aggregate()


def _step_scores(judge, q: dict, rec) -> dict[str, float]:
    """Score each step independently. Missing step => None."""
    scores: dict[str, float] = {}

    if rec.step("rewrite"):
        scores["query"] = _score(
            judge,
            task=f"Question: {q['question']}",
            candidate=rec.step("rewrite").output,
            gold=q["question"],
            rubric=QUERY_QUALITY,
            task_id=f"{q['id']}:rewrite",
        )

    retrieve_step = rec.step("retrieve_2") or rec.step("retrieve")
    if retrieve_step:
        ranks = json.loads(retrieve_step.output)
        rank = ranks.index(q["gold_doc"]) + 1 if q["gold_doc"] in ranks else 0
        scores["retrieval"] = 5.0 if rank == 1 else (3.0 if rank > 0 else 0.0)

    reason_step = rec.step("reason_2") or rec.step("reason")
    if reason_step:
        scores["reason"] = _score(
            judge,
            task=q["question"],
            candidate=reason_step.output + f"\nCited: {rec.cited_doc_ids}",
            gold=f"{q['gold_answer']} (gold doc: {q['gold_doc']})",
            rubric=REASONING_GROUNDING,
            task_id=f"{q['id']}:reason",
        )

    reflect_step = rec.step("reflect")
    if reflect_step:
        # ground truth verdict = (gold doc was in retrievals AND cited)
        truth = q["gold_doc"] in rec.retrieved_doc_ids and q["gold_doc"] in rec.cited_doc_ids
        scores["reflect"] = _score(
            judge,
            task=q["question"],
            candidate=f"verdict={reflect_step.output}",
            gold=f"verdict={'SUPPORTED' if truth else 'NEEDS_MORE'}",
            rubric=REFLECTION_CALIBRATION,
            task_id=f"{q['id']}:reflect",
        )

    scores["final"] = _score(
        judge,
        task=q["question"],
        candidate=rec.final_answer,
        gold=q["gold_answer"],
        rubric=FINAL_ANSWER,
        task_id=f"{q['id']}:final",
    )
    return scores


def main() -> int:
    pipeline = _pipeline_provider()
    judge = get_provider()
    print(f"pipeline provider: {pipeline.name}")
    print(f"judge provider:    {judge.name}\n")

    corpus = load_corpus()
    retriever = BM25(corpus)
    qs = json.loads((HERE / "questions.json").read_text())["questions"]

    cols = ("query", "retrieval", "reason", "reflect", "final")
    print(f"{'qid':<22} {'mode':<10} "
          + " ".join(f"{c:>9}" for c in cols))
    print("-" * 80)

    totals: dict[str, dict[str, float]] = {"naive": {}, "re2search": {}}
    counts: dict[str, dict[str, int]] = {"naive": {}, "re2search": {}}

    for q in qs:
        for mode, runner in (("naive", run_naive), ("re2search", run_re2search)):
            rec = runner(pipeline, retriever, q["question"])
            s = _step_scores(judge, q, rec)
            cells = [f"{s[c]:>9.2f}" if c in s else f"{'—':>9}" for c in cols]
            print(f"{q['id']:<22} {mode:<10} " + " ".join(cells))
            for c, v in s.items():
                totals[mode][c] = totals[mode].get(c, 0.0) + v
                counts[mode][c] = counts[mode].get(c, 0) + 1
        print()

    print("-" * 80)
    print(f"{'AVERAGES':<22}")
    for mode in ("naive", "re2search"):
        avgs = []
        for c in cols:
            if counts[mode].get(c, 0):
                avgs.append(f"{totals[mode][c] / counts[mode][c]:>9.2f}")
            else:
                avgs.append(f"{'—':>9}")
        print(f"{'':<22} {mode:<10} " + " ".join(avgs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
