"""Quality-gated granularity control wrapper.

Reproduction of the Agent Capsules pattern (arXiv 2605.00410). For each
pipeline step:

  - baseline mode  → prompt = question + source + every prior note (kitchen sink)
  - capsule mode   → prompt = only the agent's declared inputs (compressed)
                    then a cheap shape gate. If output looks incomplete,
                    escalate once: re-run with the full kitchen-sink prompt.

Tokens are counted on the input prompt with a 4-char-per-token estimate, which
matches well enough to expose the gap between the two modes. The point is the
relative reduction, not the absolute number.
"""
from __future__ import annotations

import json
import re

from pipeline import AgentSpec, AgentState, PIPELINE


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _baseline_prompt(agent: AgentSpec, state: AgentState) -> str:
    notes = json.dumps(state.notes, ensure_ascii=False, indent=2)
    return (
        f"Role: {agent.role}\n"
        f"Instruction: {agent.instruction}\n\n"
        f"Question: {state.question}\n\n"
        f"Source document:\n{state.source}\n\n"
        f"Prior outputs:\n{notes}\n"
    )


def _capsule_prompt(agent: AgentSpec, state: AgentState, *, full: bool) -> str:
    if full:
        return _baseline_prompt(agent, state)
    lines = [f"Role: {agent.role}", f"Instruction: {agent.instruction}", ""]
    for key in agent.inputs:
        if key == "question":
            lines.append(f"Question: {state.question}")
        elif key == "source":
            lines.append(f"Source document:\n{state.source}")
        elif key in state.notes:
            lines.append(f"{key}: {json.dumps(state.notes[key], ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def _gate_pass(agent: AgentSpec, output: str) -> bool:
    if agent.output_shape == "text":
        sentences = re.split(r"(?<=[.!?])\s+", output.strip())
        return len([s for s in sentences if s.strip()]) >= 4
    m = re.search(r"\{.*\}", output, re.S)
    if not m:
        return False
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return bool(data) and any(bool(v) for v in data.values())


def _store(agent: AgentSpec, output: str, state: AgentState) -> None:
    if agent.output_shape == "text":
        state.notes[f"{agent.name}.summary"] = output.strip()
        return
    m = re.search(r"\{.*\}", output, re.S)
    if not m:
        state.notes[f"{agent.name}.raw"] = output.strip()
        return
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        state.notes[f"{agent.name}.raw"] = output.strip()
        return
    for k, v in data.items():
        state.notes[f"{agent.name}.{k}"] = v


def run_pipeline(provider, scenario: dict, *, mode: str = "baseline") -> dict:
    if mode not in {"baseline", "capsule"}:
        raise ValueError(f"unknown mode: {mode}")

    state = AgentState(question=scenario["question"], source=scenario["source"])
    total_tokens = 0
    escalations = 0

    for agent in PIPELINE:
        if mode == "baseline":
            prompt = _baseline_prompt(agent, state)
            total_tokens += estimate_tokens(prompt)
            output = provider.complete(prompt, system=agent.role, max_tokens=512)
        else:
            prompt = _capsule_prompt(agent, state, full=False)
            total_tokens += estimate_tokens(prompt)
            output = provider.complete(prompt, system=agent.role, max_tokens=512)
            if not _gate_pass(agent, output):
                prompt = _capsule_prompt(agent, state, full=True)
                total_tokens += estimate_tokens(prompt)
                output = provider.complete(prompt, system=agent.role, max_tokens=512)
                escalations += 1

        _store(agent, output, state)

    return {
        "summary": state.notes.get("synthesizer.summary", ""),
        "tokens": total_tokens,
        "escalations": escalations,
        "notes": state.notes,
    }
