import json
import logging
import os
import urllib.error
import urllib.request
from typing import Dict, List, Optional


logger = logging.getLogger("UrbanExp")


class EmbeddingTestHTTPEmbedder:
    """
    Client for the local embedding service used in `embedding_test.py`.

    Endpoint contract:
      POST { "inputs": [ { "text": "...", "instruction": "..." }, ... ] }
      -> { "n": int, "dim": int, "embeddings": [[float]*dim, ...] }
    """

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        instruction: Optional[str] = None,
        timeout_s: float = 120.0,
        cache: bool = True,
    ):
        self.api_url = (api_url or os.getenv("EVOSHOT_EMBED_API") or "http://localhost:8000/embed").strip()
        self.instruction = (
            instruction
            or os.getenv("EVOSHOT_EMBED_INSTRUCTION")
            or "Retrieve images or text relevant to the user query."
        )
        self.timeout_s = float(os.getenv("EVOSHOT_EMBED_TIMEOUT_S", str(timeout_s)))
        self._cache_enabled = cache
        self._cache: Dict[str, List[float]] = {}
        self.dim: Optional[int] = None

    def _post(self, texts: List[str]) -> dict:
        payload = {"inputs": [{"text": t, "instruction": self.instruction} for t in texts]}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from embedder: {body}") from exc
        return json.loads(raw)

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        results: List[Optional[List[float]]] = [None] * len(texts)
        missing: List[str] = []
        missing_idx: List[int] = []

        if self._cache_enabled:
            for idx, t in enumerate(texts):
                cached = self._cache.get(t)
                if cached is not None:
                    results[idx] = cached
                else:
                    missing.append(t)
                    missing_idx.append(idx)
        else:
            missing = list(texts)
            missing_idx = list(range(len(texts)))

        if missing:
            data = self._post(missing)
            self.dim = int(data.get("dim")) if data.get("dim") is not None else self.dim
            embeds = data.get("embeddings") or []
            for t, idx, vec in zip(missing, missing_idx, embeds):
                vec_list = [float(x) for x in vec]
                results[idx] = vec_list
                if self._cache_enabled:
                    self._cache[t] = vec_list

        dim = self.dim or 2048
        final: List[List[float]] = []
        for vec in results:
            if vec is None:
                final.append([0.0] * dim)
            else:
                final.append(vec)
        return final

    def encode(self, text: str) -> List[float]:
        try:
            return self.encode_many([text])[0]
        except Exception as exc:
            dim = self.dim or 2048
            logger.error(f"Embedding failed: {exc}")
            return [0.0] * dim

