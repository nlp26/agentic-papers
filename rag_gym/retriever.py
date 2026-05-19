"""Dependency-free retriever. BM25-lite over the local corpus."""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
import json


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Standard BM25 over a small fixed corpus. k1=1.5, b=0.75."""

    def __init__(self, docs: list[dict], *, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.tokens = [_tok(d["text"]) for d in docs]
        self.lens = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lens) / max(len(self.lens), 1)
        self.df: Counter = Counter()
        for tks in self.tokens:
            for term in set(tks):
                self.df[term] += 1
        N = len(docs)
        self.idf = {
            term: math.log(1 + (N - df + 0.5) / (df + 0.5))
            for term, df in self.df.items()
        }

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _tok(query)
        scores = []
        for i, tks in enumerate(self.tokens):
            tf = Counter(tks)
            score = 0.0
            dl = self.lens[i]
            for term in q:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                f = tf[term]
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
                score += idf * num / den
            scores.append((score, i))
        scores.sort(reverse=True)
        out = []
        for sc, i in scores[:k]:
            if sc <= 0:
                continue
            out.append({**self.docs[i], "score": sc})
        return out


def load_corpus(path: str | Path = None) -> list[dict]:
    p = Path(path or Path(__file__).resolve().parent / "corpus.json")
    return json.loads(p.read_text())["docs"]
