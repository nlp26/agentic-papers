"""Re²Search: rewrite → retrieve → reason → reflect, with an optional second pass.

Two modes:
  - naive    : skip rewrite and reflect, retrieve on raw question, answer directly
  - re2search: full loop with reasoning reflection

Each step's output is captured separately so a process-supervision judge can
score individual decisions, not only the final answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from retriever import BM25


@dataclass
class StepRecord:
    name: str
    input: str
    output: str


@dataclass
class RunRecord:
    question: str
    steps: list[StepRecord] = field(default_factory=list)
    final_answer: str = ""
    cited_doc_ids: list[str] = field(default_factory=list)
    retrieved_doc_ids: list[str] = field(default_factory=list)

    def step(self, name: str) -> StepRecord | None:
        for s in self.steps:
            if s.name == name:
                return s
        return None


def _llm(provider, prompt: str, *, system: str | None = None) -> str:
    return provider.complete(prompt, system=system or "You are a helpful assistant.", max_tokens=512)


def _rewrite(provider, question: str) -> str:
    out = _llm(
        provider,
        f"Rewrite the user question into a search query (keywords only, no stopwords).\n"
        f"Reply with the query on a single line.\n\nQuestion: {question}",
    )
    return out.strip().splitlines()[0][:200]


def _reason(provider, question: str, docs: list[dict]) -> tuple[str, list[str]]:
    refs = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    out = _llm(
        provider,
        "Answer the question using only the references. Cite the [id] tags you used.\n"
        "Reply in two lines:\n"
        "ANSWER: <one or two sentences>\n"
        "CITED: <comma-separated [id] tags>\n\n"
        f"Question: {question}\n\nReferences:\n{refs}",
    )
    ans = ""
    cited: list[str] = []
    for line in out.splitlines():
        if line.lower().startswith("answer:"):
            ans = line.split(":", 1)[1].strip()
        elif line.lower().startswith("cited:"):
            cited = re.findall(r"[a-z0-9_]+", line.split(":", 1)[1].lower())
    return ans, cited


def _reflect(provider, question: str, answer: str, docs: list[dict]) -> str:
    refs = "\n".join(f"[{d['id']}] {d['text']}" for d in docs)
    out = _llm(
        provider,
        "Decide if the candidate answer is fully supported by the references.\n"
        "Reply with a single word: SUPPORTED or NEEDS_MORE.\n\n"
        f"Question: {question}\n\nCandidate answer: {answer}\n\nReferences:\n{refs}",
    )
    word = out.strip().split()[0].upper() if out.strip() else "NEEDS_MORE"
    return "SUPPORTED" if word.startswith("SUPPORT") else "NEEDS_MORE"


def run_naive(provider, retriever: BM25, question: str, k: int = 3) -> RunRecord:
    rec = RunRecord(question=question)
    docs = retriever.search(question, k=k)
    rec.retrieved_doc_ids = [d["id"] for d in docs]
    rec.steps.append(StepRecord("retrieve", question, json.dumps(rec.retrieved_doc_ids)))

    answer, cited = _reason(provider, question, docs)
    rec.steps.append(StepRecord("reason", json.dumps(rec.retrieved_doc_ids), answer))
    rec.final_answer = answer
    rec.cited_doc_ids = cited
    return rec


def run_re2search(provider, retriever: BM25, question: str, k: int = 3) -> RunRecord:
    rec = RunRecord(question=question)

    query = _rewrite(provider, question)
    rec.steps.append(StepRecord("rewrite", question, query))

    docs = retriever.search(query, k=k)
    rec.retrieved_doc_ids = [d["id"] for d in docs]
    rec.steps.append(StepRecord("retrieve", query, json.dumps(rec.retrieved_doc_ids)))

    answer, cited = _reason(provider, question, docs)
    rec.steps.append(StepRecord("reason", json.dumps(rec.retrieved_doc_ids), answer))

    verdict = _reflect(provider, question, answer, docs)
    rec.steps.append(StepRecord("reflect", answer, verdict))

    if verdict == "NEEDS_MORE":
        # one corrective pass: re-rewrite with the existing answer as context
        new_query = _rewrite(provider, f"{question} (need more evidence than: {answer})")
        rec.steps.append(StepRecord("rewrite_2", question, new_query))
        more = retriever.search(new_query, k=k)
        merged = {d["id"]: d for d in docs + more}
        docs = list(merged.values())[: k + 2]
        rec.retrieved_doc_ids = [d["id"] for d in docs]
        rec.steps.append(StepRecord("retrieve_2", new_query, json.dumps(rec.retrieved_doc_ids)))
        answer, cited = _reason(provider, question, docs)
        rec.steps.append(StepRecord("reason_2", json.dumps(rec.retrieved_doc_ids), answer))

    rec.final_answer = answer
    rec.cited_doc_ids = cited
    return rec
