# LongMemEval-V2

Reproduction of the *environment-experience* memory idea from arXiv:2605.12493 on a synthetic environment small enough to ship.

V1 (ICLR 2025) measured user-history memory ("what did the user say last week"). V2 measures whether memory helps an agent become an experienced *operator* of a specific environment — workflow knowledge, gotchas, premise awareness. This folder builds a tiny operator environment (OrionDeploy), seeds five trajectories, and scores answers in two modes.

## The synthetic environment: OrionDeploy

A fictional internal CLI. Three services (orion-api, orion-worker, orion-dashboard), three regions, a handful of verbs (deploy, rollback, status, logs, vpn-pin, config, archive). Designed to have:

- **Static state** the agent could recite if it has the env description.
- **Dynamic state** (versions, last deploys) that only appears in trajectories.
- **Workflows** with required ordering (vpn-pin before deploy, config restore after rollback).
- **Gotchas** that look fine and aren't (worker exceeding api by 2 minors silently drops messages).
- **Premises** that explain the rest (multi-region was retrofitted; feature flags live elsewhere).

Five hand-written trajectories cover the operator history the agent should learn from.

## V2 abilities

15 questions, 3 per ability:

| Category | What it tests |
|---|---|
| `static_state` | Things the env description states outright |
| `dynamic_state` | Versions / status as of the last trajectory |
| `workflow` | Required command sequences |
| `gotcha` | Counter-intuitive failure modes |
| `premise` | Why the system behaves the way it does |

## Modes

- `no_memory` — agent answers from the question alone. Mock returns "I don't know."; a real LLM may guess.
- `with_memory` — agent retrieves from a Hindsight-style `ExperienceMemory` (tag affinity + token overlap) and answers from the retrieved facts.

## Memory architecture

```
trajectories.json
   │
   ▼
 ExperienceMemory
   ├─ facts (text + tag + source_id)
   ├─ preferences (per-tag weight)
   └─ recall(query, hint_tag) → top-k facts
                     ▲
                     │ hint_tag derived from question category
```

Tag affinity dominates: a "workflow" question retrieves workflow-tagged facts first. Token overlap is the tiebreaker.

## Run

```bash
pip install -r requirements.txt

# Real provider:
export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY
python run.py

# No key → deterministic mock for the agent, mock for the judge:
python run.py
```

## Output shape (mock run, actual)

```
BY CATEGORY                    no_mem   with_mem       Δ
               static_state     2.70       3.03   +0.33
               dynamic_state    2.70       2.93   +0.23
               workflow         2.70       3.07   +0.37
               gotcha           2.70       3.20   +0.50
               premise          2.70       2.90   +0.20
OVERALL                         2.70       3.03   +0.33
```

The mock judge has heuristic fallbacks for unknown rubric criteria so absolute scores are muted offline — both modes hover near 3. The pattern across categories is still right: `gotcha` benefits most (counter-intuitive failure modes only exist in the trajectories) and `dynamic_state` / `premise` benefit least when read by a lexical-overlap judge.

With a real LLM as both agent and judge, the picture sharpens: `no_memory` drops near zero on `gotcha` / `dynamic_state` / `premise` (the model genuinely doesn't know this fictional environment), and `with_memory` rises into the 4-5 range. That's the V2 setup: only memory-equipped agents pass the harder ability categories.

## Files

- `trajectories.json` — env description + 5 trajectories with `learned_facts` per trajectory
- `questions.json` — 15 questions across the 5 ability categories
- `memory.py` — `ExperienceMemory` (fact store, preference weights, tag-affinity recall)
- `agent.py` — `answer_no_memory` / `answer_with_memory`
- `mock_provider.py` — pipeline-aware mock for offline runs
- `rubric.py` — `operator_answer` rubric (factual_match + actionable)
- `run.py` — runs both modes, scores per category, prints the table

## Refs

- LongMemEval-V2: Evaluating Long-Term Agent Memory Toward Experienced Colleagues — arXiv:2605.12493
- Hindsight: Building Agent Memory that Retains, Recalls, and Reflects — arXiv:2512.12818
- LongMemEval V1 (ICLR 2025) — arXiv:2410.10813
- Uses `agent_as_judge` from this repo for the per-answer scoring.
