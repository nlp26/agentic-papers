# agentic-papers

Read a paper, write the code, run it. One folder per paper.

Shared helpers live under `_common/` — anything reusable across papers (judge, metrics, provider wrapper) goes there so each folder stays small.

## Layout

```
agentic-papers/
  _common/                judge, providers, metrics
  agent_as_judge/         rubric scoring + CoT + calibration
  agent_capsules/         quality-gated granularity control, with token+quality chart
  rag_gym/                todo — process-supervised RAG optimization
  longmemeval_v2/         todo — environment-experience memory benchmark
  toolcua/                todo — GUI/Tool path orchestration for CUA
```

## Running

Each folder has its own `requirements.txt`.

```bash
cd <paper-folder>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` for a real run. With no key the demo uses a deterministic mock provider — useful to confirm the wiring before paying for tokens.

## Why not put this inside `architecture`

`architecture` is the pattern catalog. This is the bench where I try the techniques and keep a number next to each. Cross-reference, don't merge.

## License

Apache-2.0
