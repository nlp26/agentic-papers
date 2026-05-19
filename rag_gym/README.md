# RAG-Gym

Process-supervised RAG: every intermediate decision gets its own score, not only the final answer. Reproduction of the optimization axes from arXiv:2502.13957 on a small fixed corpus.

## Idea

Final-answer scoring tells you only whether the system worked. Process supervision tells you *where* it failed. Two agents run over the same questions:

- `naive` — retrieve on the raw question, answer directly.
- `re2search` — rewrite → retrieve → reason → reflect, with one corrective pass when reflect says `NEEDS_MORE`. This is the Re²Search shape from the paper (reasoning reflection on top of ReAct).

Each step's output is scored separately by the judge:

- `query` — is the rewritten search query sharper than the raw question?
- `retrieval` — did the retriever put the gold document at rank 1?
- `reason` — does the answer cite the gold document and match its claim?
- `reflect` — does the reflect verdict agree with whether the answer is actually grounded?
- `final` — does the final answer match the gold?

## Corpus

8 short docs about: deepagents 0.6.2, Agent Capsules, RAG-Gym, FastMCP, LongMemEval-V2, ToolCUA, Agent-as-a-Judge survey, MCP code execution. 6 questions with gold answers and gold doc ids.

## Run

```bash
pip install -r requirements.txt

# Real provider:
export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY
python run.py

# No key set → deterministic mocks for both pipeline and judge:
python run.py
```

## Output shape

```
qid                    mode           query retrieval    reason   reflect     final
q1_capsules_reduction  naive              —      5.00      5.00         —      5.00
q1_capsules_reduction  re2search       5.00      5.00      5.00      5.00      5.00
...
AVERAGES               naive              —      X.XX      X.XX         —      X.XX
                       re2search       Y.YY      Y.YY      Y.YY      Y.YY      Y.YY
```

`—` marks steps that mode doesn't execute. The naive run has no rewrite and no reflect by design — the table makes that visible.

## What process supervision buys you

- A naive run that happens to land the final answer can still have bad retrieval. Final-only scoring hides that. The `retrieval` column reveals it.
- A re2search run with a perfect retrieval but a wrong reflect verdict is a calibration bug. The `reflect` column flags it.
- Each column is independently optimizable: prompt for `rewrite`, retriever tuning for `retrieval`, prompt + model for `reason`, prompt for `reflect`. The paper's three optimization axes (prompt engineering, actor tuning, critic training) map onto specific columns.

## Files

- `corpus.json` — 8 short documents
- `questions.json` — 6 questions + gold answers + gold doc ids
- `retriever.py` — BM25-lite, no dependencies
- `re2search.py` — `run_naive` and `run_re2search` agents
- `mock_rag.py` — pipeline-aware mock provider for offline runs
- `process_rubric.py` — per-step rubrics
- `run.py` — runs both modes, scores per-step, prints the table

## Refs

- RAG-Gym: Systematic Optimization of Language Agents for Retrieval-Augmented Generation — arXiv:2502.13957
- Re²Search agent architecture — same paper
- Adaptive RAG context (2026) — process supervision is the lever that ties together the three optimization axes
- Uses `agent_as_judge` from this repo for the per-step scoring
