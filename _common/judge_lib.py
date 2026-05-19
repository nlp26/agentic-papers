"""Rubric-based judge. Provider-agnostic."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable


@dataclass
class Criterion:
    name: str
    description: str
    anchors: dict[int, str] = field(default_factory=dict)  # score -> meaning


@dataclass
class Rubric:
    name: str
    criteria: list[Criterion]
    scale: tuple[int, int] = (0, 5)

    def as_prompt_block(self) -> str:
        lines = []
        for c in self.criteria:
            lines.append(f"- {c.name}: {c.description}")
            for score, anchor in sorted(c.anchors.items()):
                lines.append(f"    {score}: {anchor}")
        return "\n".join(lines)


@dataclass
class JudgeResult:
    task_id: str
    scores: dict[str, float]
    reasons: dict[str, str]
    overall_reason: str = ""

    def aggregate(self) -> float:
        return mean(self.scores.values()) if self.scores else 0.0


_JUDGE_SYSTEM = (
    "You are a careful, calibrated evaluator. Think step by step before scoring "
    "and reply with valid JSON only."
)


def _build_prompt(task: str, candidate: str, gold: str | None, rubric: Rubric) -> str:
    lo, hi = rubric.scale
    return f"""Evaluate the candidate against the rubric. Reason privately, then reply with JSON only.

<task>
{task}
</task>

<candidate>
{candidate}
</candidate>

<gold>
{gold or ""}
</gold>

<rubric>
{rubric.as_prompt_block()}
</rubric>

Scale: {lo} (worst) to {hi} (best). One-decimal fractional scores allowed.

Reply with exactly this JSON, nothing else:
{{
  "criteria": {{
    "<criterion_name>": {{"score": <number>, "reason": "<one short sentence>"}}
  }},
  "overall_reason": "<one short sentence>"
}}
"""


def _parse(raw: str, rubric: Rubric, task_id: str) -> JudgeResult:
    # tolerate code fences and chatter
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"no JSON object in judge output for task {task_id}: {raw[:200]!r}")
    payload = json.loads(m.group(0))
    crits = payload.get("criteria", {})
    scores, reasons = {}, {}
    for c in rubric.criteria:
        entry = crits.get(c.name) or {}
        scores[c.name] = float(entry.get("score", 0.0))
        reasons[c.name] = str(entry.get("reason", ""))
    return JudgeResult(
        task_id=task_id,
        scores=scores,
        reasons=reasons,
        overall_reason=str(payload.get("overall_reason", "")),
    )


def score_task(*, provider, task: str, candidate: str, rubric: Rubric,
               gold: str | None = None, task_id: str = "task") -> JudgeResult:
    prompt = _build_prompt(task, candidate, gold, rubric)
    raw = provider.complete(prompt, system=_JUDGE_SYSTEM, max_tokens=512)
    return _parse(raw, rubric, task_id)


def score_batch(*, provider, tasks: Iterable[dict], rubric: Rubric) -> list[JudgeResult]:
    # tasks: iterable of {id, task, candidate, gold?}
    out = []
    for t in tasks:
        out.append(score_task(
            provider=provider,
            task=t["task"],
            candidate=t["candidate"],
            gold=t.get("gold"),
            rubric=rubric,
            task_id=t.get("id", "task"),
        ))
    return out


def agreement(judge_results: list[JudgeResult], human_labels: list[dict],
              *, tolerance: float = 0.5) -> dict[str, float]:
    # fraction of (task, criterion) pairs where |judge - human| <= tolerance
    by_id = {h["id"]: h["scores"] for h in human_labels}
    per_crit: dict[str, list[int]] = {}
    hits, n = 0, 0
    for jr in judge_results:
        hs = by_id.get(jr.task_id)
        if not hs:
            continue
        for crit, jscore in jr.scores.items():
            if crit not in hs:
                continue
            hit = int(abs(jscore - hs[crit]) <= tolerance)
            per_crit.setdefault(crit, []).append(hit)
            hits += hit
            n += 1
    out = {c: sum(v) / len(v) for c, v in per_crit.items() if v}
    out["overall"] = hits / n if n else 0.0
    return out


def calibrate(judge_results: list[JudgeResult], human_labels: list[dict]) -> dict[str, float]:
    # per-criterion mean bias = mean(judge - human). Subtract to calibrate.
    by_id = {h["id"]: h["scores"] for h in human_labels}
    diffs: dict[str, list[float]] = {}
    for jr in judge_results:
        hs = by_id.get(jr.task_id)
        if not hs:
            continue
        for crit, jscore in jr.scores.items():
            if crit in hs:
                diffs.setdefault(crit, []).append(jscore - hs[crit])
    return {c: mean(v) for c, v in diffs.items() if v}


def apply_calibration(judge_results: list[JudgeResult],
                      bias: dict[str, float]) -> list[JudgeResult]:
    out = []
    for jr in judge_results:
        new_scores = {
            c: max(0.0, min(5.0, s - bias.get(c, 0.0)))
            for c, s in jr.scores.items()
        }
        out.append(JudgeResult(
            task_id=jr.task_id,
            scores=new_scores,
            reasons=jr.reasons,
            overall_reason=jr.overall_reason,
        ))
    return out
