# Agent Capsules

Reproduction of the quality-gated granularity control pattern from arXiv:2605.00410. Same 5-agent due-diligence pipeline runs in two modes. The script prints the token and quality delta side by side.

## Idea

A baseline multi-agent pipeline tends to dump every prior output and the full source document into every agent's prompt. Most of that context is irrelevant per step. The capsule wrapper does two things:

- Send each agent only its declared inputs (compressed prompt).
- After each step, a cheap shape gate checks whether the output looks complete. If not, escalate once: re-run the step with the kitchen-sink prompt.

The paper reports −51% input tokens vs a hand-crafted LangGraph 14-agent CI pipeline at quality parity. This folder reproduces the pattern on a smaller 5-agent due-diligence pipeline so the idea is fully readable in one screen.

## The pipeline (5 agents)

```
scoper      → defines sub-topics
researcher  → extracts facts from the source per sub-topic
analyst     → names a risk and opportunity per sub-topic
critic      → flags any analysis unsupported by facts
synthesizer → 5-sentence summary citing strongest risk + opportunity
```

## Modes

- `baseline` — every agent receives `question + full source + all prior notes`.
- `capsule` — every agent receives only its declared inputs. If the gate fails, that step is re-run once at full granularity.

Tokens are estimated as `len(prompt) // 4` (≈ 4 chars per token). It's the relative gap that matters, not the absolute number.

## Quality side of the chart

Uses `_common/judge_lib` from the `agent_as_judge` folder. The judge scores each final summary against the gold summary with three criteria: factual grounding, coverage, risk/opportunity specificity.

## Run

```bash
pip install -r requirements.txt

# Real provider:
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
python run.py

# No key set → deterministic mock for the pipeline + mock judge:
python run.py
```

## Output shape

```
pipeline provider: mock-pipeline
judge provider:    mock

scenario              base_tok  caps_tok    Δtok%  esc  q_base  q_caps     Δq
--------------------------------------------------------------------------------
acme_security             X         Y     -NN.N%    0    X.XX    Y.YY  +0.00
microvolt_finance         X         Y     -NN.N%    0    X.XX    Y.YY  +0.00
deltadrop_legal           X         Y     -NN.N%    0    X.XX    Y.YY  +0.00
--------------------------------------------------------------------------------
TOTAL                     X         Y     -NN.N%         X.XX    Y.YY  +0.00
```

With the mock the pipeline returns identical canned outputs in both modes, so the quality delta is zero by construction — the demo shows the token reduction. With a real model both sides become meaningful.

## What's mock-only vs real-only

| Signal | Mock | Real |
|---|---|---|
| Token gap baseline vs capsule | meaningful | meaningful |
| Quality gap baseline vs capsule | constant (mock returns the same text) | meaningful |
| Gate escalations | rarely fire (canned output passes) | reflects real failures |

## Files

- `pipeline.py` — agent specs and pipeline state
- `capsule.py` — baseline and capsule prompt builders + gate + runner
- `mock_pipeline.py` — pipeline-aware mock provider for offline runs
- `scenarios.json` — 3 due-diligence scenarios with gold summaries
- `rubric.py` — `due_diligence_summary` rubric used by the judge
- `run.py` — runs everything and prints the table

## Refs

- Agent Capsules: Quality-Gated Granularity Control — arXiv:2605.00410
- Uses `agent_as_judge` from this repo for the quality side
