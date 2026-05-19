"""Mock provider tuned for the LongMemEval-V2 demo.

Detects whether the prompt has a 'Relevant memories:' block. If so it synthesizes
an answer from the listed facts. If not, it returns a confident 'I don't know.'
This makes the no-memory vs with-memory gap visible offline; with a real LLM the
same prompts produce richer answers in both modes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_FACT_LINE = re.compile(r"^- \[(?P<tag>[a-z_]+)\] (?P<text>.+?)(?:\s+\(source:.*)?$", re.M)


@dataclass
class MockMemoryProvider:
    name: str = "mock-memory"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 200) -> str:
        if "Relevant memories:" not in prompt:
            return "I don't know."
        block = prompt.split("Relevant memories:", 1)[1]
        block = block.split("Question:", 1)[0]
        facts = [(m.group("tag"), m.group("text").strip()) for m in _FACT_LINE.finditer(block)]
        if not facts:
            return "I don't know."
        # Compose: lead with the highest-affinity fact, optionally tack on one more.
        primary = facts[0][1]
        if len(facts) > 1 and facts[1][0] in {"workflow", "premise"}:
            return f"{primary} {facts[1][1]}"
        return primary
