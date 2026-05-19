"""Mock provider tuned for the due-diligence pipeline. Returns canned JSON shapes
that pass the gate, so the demo runs offline. Token accounting reflects real
prompt size; output quality is constant and not a meaningful signal in mock mode.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class MockPipelineProvider:
    name: str = "mock-pipeline"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        # detect agent by keywords in the instruction text
        p = prompt.lower()
        if "sub_topics" in p and "list 3" in p:
            return json.dumps({"sub_topics": ["governance", "financials", "compliance"]})
        if '"facts"' in prompt or "factual bullet points" in p:
            return json.dumps({"facts": {
                "governance": ["Board has 7 members.", "Quarterly external audit since 2024."],
                "financials": ["Revenue +18% YoY.", "Operating margin 12%."],
                "compliance": ["Two open regulatory inquiries.", "GDPR DPO appointed 2025."],
            }})
        if '"analysis"' in prompt or "risk and one opportunity" in p:
            return json.dumps({"analysis": {
                "governance": {"risk": "Concentrated voting power.", "opportunity": "Independent audit trail."},
                "financials": {"risk": "Margin compression next FY.", "opportunity": "Growth above sector mean."},
                "compliance": {"risk": "Outcome of open inquiries unknown.", "opportunity": "Mature privacy posture."},
            }})
        if '"weak_points"' in prompt or "challenge the analysis" in p:
            return json.dumps({"weak_points": []})
        if "5-sentence summary" in p or "5-sentence" in p:
            return (
                "The review covers governance, financials, and compliance. "
                "The strongest risk is the unresolved regulatory inquiries flagged in compliance. "
                "The strongest opportunity is revenue growth materially above sector mean. "
                "Governance shows mature controls but voting power is concentrated. "
                "Overall posture is acceptable subject to monitoring the open inquiries."
            )
        # fallback: empty json so gate fails and forces an escalation
        return "{}"
