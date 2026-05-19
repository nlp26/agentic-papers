"""5-agent due-diligence pipeline. Framework-agnostic: an agent is a callable
specification (role, instruction, declared inputs) that takes state and adds
output to it. The capsule wrapper in capsule.py treats this as its target.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentSpec:
    name: str
    role: str
    instruction: str
    inputs: list[str]      # state keys this agent needs in capsule mode
    output_shape: str      # "json" or "text"


@dataclass
class AgentState:
    question: str
    source: str
    notes: dict = field(default_factory=dict)


PIPELINE: list[AgentSpec] = [
    AgentSpec(
        name="scoper",
        role="You define the scope of a due-diligence review.",
        instruction=(
            "Given the question, list 3 short sub-topics that should be investigated. "
            'Return JSON {"sub_topics": ["...", "...", "..."]}.'
        ),
        inputs=["question"],
        output_shape="json",
    ),
    AgentSpec(
        name="researcher",
        role="You extract facts from the source document.",
        instruction=(
            "For each sub-topic, extract at most 2 short factual bullet points "
            "from the source document. "
            'Return JSON {"facts": {"<sub_topic>": ["...", "..."], ...}}.'
        ),
        inputs=["question", "source", "scoper.sub_topics"],
        output_shape="json",
    ),
    AgentSpec(
        name="analyst",
        role="You analyze risks and opportunities.",
        instruction=(
            "From the facts, identify one risk and one opportunity per sub-topic. "
            'Return JSON {"analysis": {"<sub_topic>": {"risk": "...", "opportunity": "..."}, ...}}.'
        ),
        inputs=["researcher.facts"],
        output_shape="json",
    ),
    AgentSpec(
        name="critic",
        role="You challenge the analysis quality.",
        instruction=(
            "Return any sub-topic whose risk or opportunity is unsupported by the facts. "
            'Return JSON {"weak_points": ["..."]}. Empty list if all sound.'
        ),
        inputs=["researcher.facts", "analyst.analysis"],
        output_shape="json",
    ),
    AgentSpec(
        name="synthesizer",
        role="You write the final due-diligence summary.",
        instruction=(
            "Write a 5-sentence summary covering the question. Cite the strongest risk "
            "and the strongest opportunity. Plain text only, no JSON."
        ),
        inputs=["question", "researcher.facts", "analyst.analysis", "critic.weak_points"],
        output_shape="text",
    ),
]
