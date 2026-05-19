# Agent-as-a-Judge

Rubric-based LLM evaluator with chain-of-thought scoring and bias calibration against human labels.

## Idea

Three things matter for a useful judge:

- Score named criteria with anchors, not one opaque number.
- Make the judge reason before scoring (CoT). Single biggest reliability lever.
- When you have human labels, measure agreement and subtract per-criterion bias.

That's the whole pipeline. The reusable bits sit in `../_common/judge_lib.py` so the next paper folder can import them.

## Papers

- Agent-as-a-Judge — arXiv:2410.10934
- Survey on Agent-as-a-Judge — arXiv:2601.05111
- Auto-Eval Judge — arXiv:2508.05508
- Rubric-based evals + IRT direction — Adnan Masood, 2026

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Real provider:
export ANTHROPIC_API_KEY=...
python run.py

# No key set → deterministic mock runs anyway:
python run.py
```

## Output shape

10 questions, two candidates each (strong / weak), human labels for all 20.

```
provider: mock
rubric:   qa_short_answer

scores (uncalibrated):
  task_01_strong  faithfulness=4.8 completeness=5.0 reasoning=4.3   -> 4.70
  task_01_weak    faithfulness=1.8 completeness=1.8 reasoning=2.3   -> 1.97
  ...

agreement vs human (tol=0.5):
  faithfulness   0.10
  completeness   0.45
  reasoning      0.50
  overall        0.35

per-criterion bias (judge - human):
  faithfulness   -1.33
  completeness   -0.11
  reasoning      +0.36

agreement after calibration:
  overall        0.47
```

Numbers above are from the mock provider — useful to see the calibration step working. Real model lands the un-calibrated overall in the 0.80+ range.

## Files

- `rubrics.py` — rubric definitions, extend here
- `sample_tasks.json` — tasks + candidates + human labels
- `run.py` — end-to-end CLI demo
- `agent_as_judge.ipynb` — same pipeline, cell-by-cell
