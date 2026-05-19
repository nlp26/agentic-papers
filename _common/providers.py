"""Provider wrapper. Anything with .complete(prompt, system=, max_tokens=) is a provider."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Protocol


class Provider(Protocol):
    name: str

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str: ...


@dataclass
class AnthropicProvider:
    model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    name: str = "anthropic"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        from anthropic import Anthropic

        client = Anthropic()
        msg = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system or "You are a careful evaluator.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text"))


@dataclass
class OpenAIProvider:
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    name: str = "openai"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system or "You are a careful evaluator."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


@dataclass
class MockProvider:
    # Offline stand-in. Scores from lexical overlap, length ratio, and a reasoning keyword.
    # Returns the JSON shape the judge prompt asks for so the rest of the pipeline runs unchanged.
    seed: int = 0
    name: str = "mock"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        gold = _between(prompt, "<gold>", "</gold>")
        candidate = _between(prompt, "<candidate>", "</candidate>")
        rubric_block = _between(prompt, "<rubric>", "</rubric>")
        criteria = re.findall(r"-\s*([a-z_]+):", rubric_block or "")

        g, c = _toks(gold), _toks(candidate)
        overlap = len(g & c) / max(len(g), 1)
        length_ratio = min(len(c) / max(len(g), 1), 1.5)
        has_reasoning = bool(re.search(r"\b(because|since|therefore|so)\b", candidate, re.I))

        defaults = {
            "faithfulness": round(5 * overlap, 1),
            "completeness": round(5 * min(length_ratio, 1.0), 1),
            "reasoning": 4.0 if has_reasoning else 2.0,
        }
        # tiny deterministic jitter so outputs aren't suspiciously identical
        h = int(hashlib.sha1((candidate or "x").encode()).hexdigest(), 16) % 7
        jitter = (h - 3) * 0.1

        out = {"criteria": {}, "overall_reason": "mock heuristic scoring"}
        for crit in criteria or list(defaults.keys()):
            base = defaults.get(crit, 3.0)
            score = max(0.0, min(5.0, round(base + jitter, 1)))
            out["criteria"][crit] = {
                "score": score,
                "reason": f"mock: overlap={overlap:.2f}, length_ratio={length_ratio:.2f}",
            }
        return json.dumps(out)


def _between(text: str, a: str, b: str) -> str:
    m = re.search(re.escape(a) + r"(.*?)" + re.escape(b), text, re.S)
    return (m.group(1) if m else "").strip()


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2}


def get_provider(name: str | None = None) -> Provider:
    # Auto-pick: anthropic if key, else openai if key, else mock.
    if name is None:
        if os.getenv("ANTHROPIC_API_KEY"):
            name = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            name = "openai"
        else:
            name = "mock"
    name = name.lower()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "mock":
        return MockProvider()
    raise ValueError(f"unknown provider: {name}")
