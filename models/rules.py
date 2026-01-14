import hashlib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

import numpy as np


def _norm_rule_text(text: str) -> str:
    cleaned = (text or "").strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _stable_rule_id(norm_text: str) -> str:
    h = hashlib.md5(norm_text.encode("utf-8")).hexdigest()[:10]
    return f"rule_{h}"


@dataclass
class Rule:
    id: str
    text: str
    norm: str
    created_ts: int
    count: int = 1
    sources: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None


class RuleBank:
    """
    A lightweight "semantic memory" store: short reusable rules extracted from Teacher feedback.

    This is still non-parametric learning (no fine-tuning). The rules are injected into the
    Student prompt even when example retrieval k=0, enabling more global generalization.
    """

    def __init__(self, *, embedder: Optional[Any] = None):
        self._rules: List[Rule] = []
        self._by_norm: dict[str, Rule] = {}
        self._embedder = embedder

    @property
    def rules(self) -> List[Rule]:
        return list(self._rules)

    def add(self, text: str, *, source_id: Optional[str] = None) -> Optional[Rule]:
        raw = (text or "").strip()
        if not raw:
            return None

        norm = _norm_rule_text(raw)
        if not norm:
            return None

        existing = self._by_norm.get(norm)
        if existing is not None:
            existing.count += 1
            if source_id and source_id not in existing.sources:
                existing.sources.append(source_id)
            return existing

        rid = _stable_rule_id(norm)
        rule = Rule(
            id=rid,
            text=raw,
            norm=norm,
            created_ts=int(time.time()),
            count=1,
            sources=[source_id] if source_id else [],
        )
        if self._embedder is not None:
            try:
                rule.embedding = self._embedder.encode(f"Rule: {rule.text}")
            except Exception:
                rule.embedding = None

        self._rules.append(rule)
        self._by_norm[norm] = rule
        return rule

    def _ensure_embeddings(self) -> None:
        if self._embedder is None:
            return
        missing: List[Rule] = [r for r in self._rules if r.embedding is None]
        if not missing:
            return
        try:
            texts = [f"Rule: {r.text}" for r in missing]
            vecs = self._embedder.encode_many(texts) if hasattr(self._embedder, "encode_many") else [self._embedder.encode(t) for t in texts]
            for r, v in zip(missing, vecs):
                r.embedding = v
        except Exception:
            return

    def select(self, *, query_text: str, k: int) -> List[Rule]:
        if k <= 0 or not self._rules:
            return []
        k = min(k, len(self._rules))

        self._ensure_embeddings()
        have_emb = any(r.embedding is not None for r in self._rules)
        if self._embedder is None or not have_emb:
            ranked = sorted(self._rules, key=lambda r: (r.count, r.created_ts), reverse=True)
            return ranked[:k]

        try:
            q = np.asarray(self._embedder.encode(f"Query: {query_text}"), dtype=np.float32)
        except Exception:
            ranked = sorted(self._rules, key=lambda r: (r.count, r.created_ts), reverse=True)
            return ranked[:k]

        qn = float(np.linalg.norm(q)) or 1e-9
        qn_vec = q / qn

        mat = []
        candidates: List[Rule] = []
        for r in self._rules:
            if r.embedding is None:
                continue
            mat.append(r.embedding)
            candidates.append(r)

        if not candidates:
            ranked = sorted(self._rules, key=lambda r: (r.count, r.created_ts), reverse=True)
            return ranked[:k]

        mat_arr = np.asarray(mat, dtype=np.float32)
        vn = np.linalg.norm(mat_arr, axis=1)
        vn[vn == 0] = 1e-9
        mat_norm = mat_arr / vn[:, None]
        sims = mat_norm.dot(qn_vec)

        idxs = np.argsort(sims)[::-1][:k]
        return [candidates[int(i)] for i in idxs]

    @staticmethod
    def to_prompt_block(rules: Iterable[Rule]) -> str:
        lines = []
        for idx, r in enumerate(rules, start=1):
            text = (r.text or "").strip()
            if not text:
                continue
            lines.append(f"{idx}) {text}")
        return "\n".join(lines).strip()

