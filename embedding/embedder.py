import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


logger = logging.getLogger("UrbanExp")


_WIN_ABS_PATH_RE = re.compile(r"^[A-Za-z]:[\\\\/]")


def _normalize_image_ref(image_ref: str) -> str:
    """
    Normalize image references before sending them to the embedding service.

    Supported:
      - URLs / data URLs: pass through.
      - Windows absolute paths -> optional WSL /mnt/<drive>/... conversion
        (set EVOSHOT_EMBED_IMAGE_PATH_STYLE=wsl).
    """
    if not image_ref:
        return image_ref

    ref = str(image_ref).strip()
    lower = ref.lower()
    if lower.startswith(("http://", "https://", "data:image/")):
        return ref

    style = (os.getenv("EVOSHOT_EMBED_IMAGE_PATH_STYLE") or "windows").strip().lower()
    if style not in {"wsl", "wsl_mnt", "wsl-mnt"}:
        return ref

    if not _WIN_ABS_PATH_RE.match(ref):
        return ref

    drive = ref[0].lower()
    rest = ref[2:].lstrip("\\/").replace("\\", "/")
    return f"/mnt/{drive}/{rest}"


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

    def _cache_key(self, item: Dict[str, Any]) -> str:
        return json.dumps(item, ensure_ascii=False, sort_keys=True)

    def _post_inputs(self, inputs: List[Dict[str, Any]]) -> dict:
        normalized: List[Dict[str, Any]] = []
        for item in inputs:
            row: Dict[str, Any] = {"instruction": item.get("instruction") or self.instruction}
            if item.get("text") is not None:
                row["text"] = str(item.get("text"))
            if item.get("image") is not None:
                row["image"] = _normalize_image_ref(str(item.get("image")))
            normalized.append(row)

        payload = {"inputs": normalized}
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

    def encode_inputs(self, inputs: List[Dict[str, Any]]) -> List[List[float]]:
        """
        Multimodal embedding API client.

        Each input item can include:
          - {"text": "..."}
          - {"image": "/path/to/img.png"}
          - {"image": "...", "text": "..."}  (fused embedding)
        """
        if not inputs:
            return []

        results: List[Optional[List[float]]] = [None] * len(inputs)
        missing: List[Dict[str, Any]] = []
        missing_idx: List[int] = []

        if self._cache_enabled:
            for idx, item in enumerate(inputs):
                norm_image = _normalize_image_ref(str(item.get("image"))) if item.get("image") is not None else None
                key = self._cache_key({"image": norm_image, "text": item.get("text"), "instruction": item.get("instruction") or self.instruction})
                cached = self._cache.get(key)
                if cached is not None:
                    results[idx] = cached
                else:
                    if norm_image is None:
                        missing.append(item)
                    else:
                        missing.append({"image": norm_image, "text": item.get("text"), "instruction": item.get("instruction")})
                    missing_idx.append(idx)
        else:
            missing = list(inputs)
            missing_idx = list(range(len(inputs)))

        if missing:
            data = self._post_inputs(missing)
            self.dim = int(data.get("dim")) if data.get("dim") is not None else self.dim
            embeds = data.get("embeddings") or []
            for item, idx, vec in zip(missing, missing_idx, embeds):
                vec_list = [float(x) for x in vec]
                results[idx] = vec_list
                if self._cache_enabled:
                    key = self._cache_key({"image": item.get("image"), "text": item.get("text"), "instruction": item.get("instruction") or self.instruction})
                    self._cache[key] = vec_list

        dim = self.dim or 2048
        final: List[List[float]] = []
        for vec in results:
            if vec is None:
                final.append([0.0] * dim)
            else:
                final.append(vec)
        return final

    def encode_many(self, texts: List[str]) -> List[List[float]]:
        return self.encode_inputs([{"text": t} for t in texts])

    def encode(self, text: str) -> List[float]:
        try:
            return self.encode_many([text])[0]
        except Exception as exc:
            dim = self.dim or 2048
            logger.error(f"Embedding failed: {exc}")
            return [0.0] * dim

    def encode_image_path(self, image_path: str) -> List[float]:
        try:
            return self.encode_inputs([{"image": image_path}])[0]
        except Exception as exc:
            dim = self.dim or 2048
            logger.error(f"Image embedding failed: {exc}")
            return [0.0] * dim

    def encode_image_text(self, image_path: str, text: str) -> List[float]:
        try:
            return self.encode_inputs([{"image": image_path, "text": text}])[0]
        except Exception as exc:
            dim = self.dim or 2048
            logger.error(f"Image+text embedding failed: {exc}")
            return [0.0] * dim
