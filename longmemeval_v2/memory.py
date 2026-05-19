"""Experience memory — Hindsight-style. Fact store with tags, preference weights
per tag, and recall conditioned on the current task's tag affinity.

Built to back the LongMemEval-V2 abilities: static state recall, dynamic state
tracking, workflow knowledge, gotchas, premise awareness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2}


@dataclass
class Fact:
    tag: str
    text: str
    source_id: str


@dataclass
class ExperienceMemory:
    """Three-layer recall: tag affinity > token overlap > stable order tiebreak."""
    facts: list[Fact] = field(default_factory=list)
    preferences: dict[str, float] = field(default_factory=dict)
    env_lines: list[str] = field(default_factory=list)

    @classmethod
    def from_trajectories(cls, path: str | Path) -> "ExperienceMemory":
        data = json.loads(Path(path).read_text())
        mem = cls()
        env = data.get("environment", {})
        if env.get("description"):
            mem.env_lines.append(env["description"])
        for line in env.get("premises", []):
            mem.facts.append(Fact(tag="premise", text=line, source_id="environment"))
        for traj in data.get("trajectories", []):
            for f in traj.get("learned_facts", []):
                mem.facts.append(Fact(tag=f["tag"], text=f["text"], source_id=traj["id"]))
        return mem

    def reinforce(self, tag: str, weight: float = 1.0) -> None:
        self.preferences[tag] = self.preferences.get(tag, 0.0) + weight

    def recall(self, *, query: str, hint_tag: str | None = None, k: int = 4) -> list[Fact]:
        q_toks = _tokens(query)
        scored: list[tuple[float, int, Fact]] = []
        for i, f in enumerate(self.facts):
            base = self.preferences.get(f.tag, 0.0)
            if hint_tag and f.tag == hint_tag:
                base += 2.0
            overlap = len(q_toks & _tokens(f.text)) / max(len(q_toks), 1)
            scored.append((base + overlap * 5.0, i, f))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [f for _, _, f in scored[:k]]

    def env_context(self) -> str:
        return "\n".join(self.env_lines)
