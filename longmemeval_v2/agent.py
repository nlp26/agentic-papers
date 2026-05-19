"""Two modes:
  - no_memory : agent answers from question only (plus a one-line system role).
  - with_memory : agent retrieves from ExperienceMemory and answers from facts.

Designed so the provider-agnostic mock or any LLM produces meaningfully
different outputs across the two modes.
"""
from __future__ import annotations

from memory import ExperienceMemory


SYSTEM_PROMPT = (
    "You are an on-call operator for the OrionDeploy CLI. Answer in 1-2 sentences. "
    "If you do not have evidence, say 'I don't know.' Do not invent facts."
)


def _prompt_no_memory(question: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nQuestion: {question}\n\nAnswer:"


def _prompt_with_memory(question: str, *, env: str, facts: list) -> str:
    lines = [SYSTEM_PROMPT, "", "Environment:", env, "", "Relevant memories:"]
    for f in facts:
        lines.append(f"- [{f.tag}] {f.text}  (source: {f.source_id})")
    lines.extend(["", f"Question: {question}", "", "Answer:"])
    return "\n".join(lines)


def answer_no_memory(provider, question: str) -> str:
    return provider.complete(_prompt_no_memory(question), system=SYSTEM_PROMPT, max_tokens=200).strip()


def answer_with_memory(provider, memory: ExperienceMemory, question: str, *, hint_tag: str | None = None) -> str:
    facts = memory.recall(query=question, hint_tag=hint_tag)
    prompt = _prompt_with_memory(question, env=memory.env_context(), facts=facts)
    return provider.complete(prompt, system=SYSTEM_PROMPT, max_tokens=200).strip()
