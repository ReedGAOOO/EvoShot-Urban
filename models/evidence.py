import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .student import robust_json_parser


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
    """
    Encode an image as a data URL for vision evidence extraction.

    To keep calls fast, we downsize images by default:
      - EVOSHOT_EVIDENCE_IMAGE_MAX_SIDE (default 640)
      - EVOSHOT_EVIDENCE_IMAGE_JPEG_QUALITY (default 85)
    """
    max_side = int(os.getenv("EVOSHOT_EVIDENCE_IMAGE_MAX_SIDE", "640"))
    quality = int(os.getenv("EVOSHOT_EVIDENCE_IMAGE_JPEG_QUALITY", "85"))

    try:
        from PIL import Image

        img = Image.open(image_path)
        img.load()
        if img.mode not in {"RGB", "L"}:
            img = img.convert("RGB")

        w, h = img.size
        scale = max(w, h) / float(max_side) if max_side > 0 else 1.0
        if scale > 1.0:
            new_w = max(1, int(round(w / scale)))
            new_h = max(1, int(round(h / scale)))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        buf = BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}", "image/jpeg"
    except Exception:
        mime, _ = mimetypes.guess_type(image_path)
        if not mime:
            mime = "image/jpeg"
        with open(image_path, "rb") as image_file:
            b64 = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}", mime


@dataclass(frozen=True)
class EvidenceResult:
    facts: List[str]
    uncertainties: List[str]
    raw: str

    def to_prompt_block(self) -> str:
        lines: List[str] = []
        for idx, f in enumerate(self.facts or [], start=1):
            s = str(f or "").strip()
            if s:
                lines.append(f"{idx}) {s}")
        if self.uncertainties:
            lines.append("Uncertain:")
            for u in self.uncertainties:
                s = str(u or "").strip()
                if s:
                    lines.append(f"- {s}")
        return "\n".join(lines).strip()


class LocalVisionEvidenceExtractor:
    """
    Extract a short list of concrete, visibly grounded facts from an image.

    This is a "scaffolding" call to the SAME Student model endpoint (no fine-tuning),
    used to reduce hallucination by giving the scorer a set of candidate observations.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        max_tokens: int = 220,
        cache_path: Optional[str] = None,
    ):
        self.url = (base_url or os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip()
        self.model = (model or os.getenv("EVOSHOT_EVIDENCE_MODEL") or os.getenv("EVOSHOT_LLM_MODEL") or "").strip()
        if not self.model:
            self.model = "qwen3-vl-30b-a3b-thinking"

        self.timeout_s = float(os.getenv("EVOSHOT_EVIDENCE_TIMEOUT_S", str(timeout_s)))
        self.max_tokens = int(os.getenv("EVOSHOT_EVIDENCE_MAX_TOKENS", str(max_tokens)))
        self.version = (os.getenv("EVOSHOT_EVIDENCE_VERSION") or "v1").strip()
        self.use_tools = (os.getenv("EVOSHOT_EVIDENCE_USE_TOOLS") or "auto").strip().lower()
        self.tool_choice = (os.getenv("EVOSHOT_EVIDENCE_TOOL_CHOICE") or "required").strip()

        root = Path(__file__).resolve().parent.parent
        data_dir = Path(os.getenv("EVOSHOT_DATA_DIR", str(root / "data"))).resolve()
        default_cache = data_dir / "cache" / "evidence_cache.jsonl"
        self.cache_path = Path(os.getenv("EVOSHOT_EVIDENCE_CACHE_PATH", cache_path or str(default_cache))).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, EvidenceResult] = {}
        self._load_cache()

    def _cache_key(self, image_path: str) -> str:
        try:
            p = Path(image_path)
            if p.exists():
                return f"{self.version}::{self.model}::{p.resolve()}"
        except Exception:
            pass
        return f"{self.version}::{self.model}::{image_path}"

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = str(obj.get("key") or "")
                if not key:
                    continue
                facts = obj.get("facts") or []
                uncertainties = obj.get("uncertainties") or []
                raw = str(obj.get("raw") or "")
                res = EvidenceResult(
                    facts=[str(x) for x in facts if str(x).strip()],
                    uncertainties=[str(x) for x in uncertainties if str(x).strip()],
                    raw=raw,
                )
                self._cache[key] = res
        except Exception:
            return

    def _save_cache_row(self, *, key: str, res: EvidenceResult) -> None:
        row = {
            "key": key,
            "facts": list(res.facts),
            "uncertainties": list(res.uncertainties),
            "raw": res.raw,
            "ts": int(time.time()),
            "model": self.model,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _tools_spec(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_evidence",
                    "description": "Submit concrete, visibly grounded facts observed in the image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "facts": {"type": "array", "items": {"type": "string"}},
                            "uncertainties": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["facts", "uncertainties"],
                    },
                },
            }
        ]

    @staticmethod
    def _extract_tool_arguments(response_json: Dict[str, Any]) -> Optional[str]:
        try:
            message = response_json["choices"][0]["message"]
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                return None
            return tool_calls[0]["function"]["arguments"]
        except Exception:
            return None

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        try:
            return response_json["choices"][0]["message"].get("content") or ""
        except Exception:
            return ""

    def extract(self, image_path: str) -> EvidenceResult:
        key = self._cache_key(image_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        data_url, _mime = _encode_image_data_url(image_path)

        system = (
            "You are a careful visual observer.\n"
            "Return ONLY a single valid JSON object (or a tool call) with concrete facts.\n"
            "Do NOT infer hidden causes. If unsure, put it under uncertainties.\n"
        )
        prompt = """
Extract 5-8 short, concrete, visually grounded facts from the image.

Rules:
- Each fact must be directly visible (objects, lighting, road condition, crowd level, trash, graffiti).
- Do NOT guess motion, intent, or unseen items.
- If a detail is uncertain, put it in uncertainties, not facts.

Return JSON:
{
  "facts": ["...", "..."],
  "uncertainties": ["...", "..."]
}
""".strip()

        use_tools = self.use_tools in {"1", "true", "yes", "on", "required"}
        if self.use_tools == "auto":
            use_tools = True

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if use_tools:
            payload["tools"] = self._tools_spec()
            payload["tool_choice"] = self.tool_choice

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from evidence extractor: {body}") from exc

        resp_json = json.loads(raw)
        tool_args = self._extract_tool_arguments(resp_json)
        content = tool_args if tool_args is not None else self._extract_content(resp_json)

        parsed = robust_json_parser(content)
        facts = parsed.get("facts") or []
        uncertainties = parsed.get("uncertainties") or []
        res = EvidenceResult(
            facts=[str(x) for x in facts if str(x).strip()],
            uncertainties=[str(x) for x in uncertainties if str(x).strip()],
            raw=content,
        )
        self._cache[key] = res
        try:
            self._save_cache_row(key=key, res=res)
        except Exception:
            pass
        return res

