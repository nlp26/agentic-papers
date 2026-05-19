"""Mock provider tuned for the rag_gym pipeline. Detects which step it's in
by keyword and returns a deterministic plausible response. Token budget is
not modeled here — the point of this folder is the process-supervision idea.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_STOPWORDS = {
    "what", "which", "does", "do", "is", "are", "the", "a", "an", "of", "in",
    "and", "or", "to", "on", "for", "with", "how", "by", "that", "this",
    "did", "not", "as", "than", "over", "from",
}


@dataclass
class MockRagProvider:
    name: str = "mock-rag"

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512) -> str:
        p = prompt.lower()
        if "rewrite the user question into a search query" in p:
            qmatch = re.search(r"question:\s*(.+)", prompt, re.I)
            question = qmatch.group(1).strip() if qmatch else ""
            kw = [w for w in re.findall(r"[a-z0-9_-]+", question.lower()) if w not in _STOPWORDS and len(w) > 2]
            return " ".join(kw[:6]) or question
        if "answer the question using only the references" in p:
            refs = re.findall(r"\[([a-z0-9_]+)\]\s*([^\[]+)", prompt)
            qmatch = re.search(r"question:\s*(.+?)\n", prompt, re.I)
            question = (qmatch.group(1) if qmatch else "").lower()
            qkw = {w for w in re.findall(r"[a-z0-9]+", question) if w not in _STOPWORDS and len(w) > 2}
            best_id, best_text, best_overlap = "", "", -1
            for doc_id, body in refs:
                body_words = set(re.findall(r"[a-z0-9]+", body.lower()))
                overlap = len(qkw & body_words)
                if overlap > best_overlap:
                    best_overlap, best_id, best_text = overlap, doc_id, body
            sentences = re.split(r"(?<=[.!?])\s+", best_text.strip())
            answer = " ".join(sentences[:2])[:300]
            return f"ANSWER: {answer}\nCITED: [{best_id}]"
        if "decide if the candidate answer is fully supported" in p:
            return "SUPPORTED"
        return ""
