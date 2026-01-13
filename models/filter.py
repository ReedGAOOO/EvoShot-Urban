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
from typing import Any, Dict, Optional, Tuple, List

from .student import robust_json_parser


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
    """
    Encode an image as a data URL for vision calls.

    To speed up filtering on large datasets, this function downsizes images by default.
    Control via:
      - EVOSHOT_FILTER_IMAGE_MAX_SIDE (default 640)
      - EVOSHOT_FILTER_IMAGE_JPEG_QUALITY (default 85)
    """
    max_side = int(os.getenv("EVOSHOT_FILTER_IMAGE_MAX_SIDE", "640"))
    quality = int(os.getenv("EVOSHOT_FILTER_IMAGE_JPEG_QUALITY", "85"))

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


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "yes", "y", "1"}:
            return True
        if v in {"false", "no", "n", "0"}:
            return False
    return None


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class UrbanFilterResult:
    is_urban: bool
    confidence: float
    caption: str
    raw: str


class StudentUrbanFilter:
    """
    Use the (local) Student VLM to filter out non-urban/noisy images before running experiments.

    Returns a conservative decision: if unsure, it should be False.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
        max_tokens: int = 200,
        cache_path: Optional[str] = None,
    ):
        self.url = (base_url or os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip()
        self.model = (model or os.getenv("EVOSHOT_FILTER_MODEL") or os.getenv("EVOSHOT_LLM_MODEL") or "").strip()
        if not self.model:
            self.model = "qwen3-vl-30b-a3b-thinking"
        self.timeout_s = float(os.getenv("EVOSHOT_FILTER_TIMEOUT_S", str(timeout_s)))
        self.max_tokens = int(os.getenv("EVOSHOT_FILTER_MAX_TOKENS", str(max_tokens)))
        self.version = (os.getenv("EVOSHOT_FILTER_VERSION") or "v2").strip()
        self.use_tools = (os.getenv("EVOSHOT_FILTER_USE_TOOLS") or "auto").strip().lower()
        self.tool_choice = (os.getenv("EVOSHOT_FILTER_TOOL_CHOICE") or "required").strip()

        root = Path(__file__).resolve().parent.parent
        data_dir = Path(os.getenv("EVOSHOT_DATA_DIR", str(root / "data"))).resolve()
        default_cache = data_dir / "cache" / "urban_filter_cache.jsonl"
        self.cache_path = Path(os.getenv("EVOSHOT_FILTER_CACHE_PATH", cache_path or str(default_cache))).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._cache: Dict[str, UrbanFilterResult] = {}
        self._load_cache()

    def _tools_spec(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_filter",
                    "description": "Return whether the image is an outdoor urban built environment, plus a short caption.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_urban": {"type": "boolean"},
                            "confidence": {"type": "number"},
                            "caption": {"type": "string"},
                        },
                        "required": ["is_urban", "confidence", "caption"],
                    },
                },
            }
        ]

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
                res = UrbanFilterResult(
                    is_urban=bool(obj.get("is_urban")),
                    confidence=float(obj.get("confidence") or 0.0),
                    caption=str(obj.get("caption") or ""),
                    raw=str(obj.get("raw") or ""),
                )
                self._cache[key] = res
        except Exception:
            return

    def _cache_key(self, image_path: str) -> str:
        try:
            p = Path(image_path)
            if p.exists():
                return f"{self.version}::{self.model}::{p.resolve()}"
        except Exception:
            pass
        return f"{self.version}::{self.model}::{image_path}"

    def _save_cache_row(self, *, key: str, res: UrbanFilterResult) -> None:
        row = {
            "key": key,
            "is_urban": bool(res.is_urban),
            "confidence": float(res.confidence),
            "caption": res.caption,
            "raw": res.raw,
            "ts": int(time.time()),
            "model": self.model,
        }
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def classify(self, image_path: str) -> UrbanFilterResult:
        key = self._cache_key(image_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        data_url, _mime = _encode_image_data_url(image_path)
        system = (
            "You are a strict image classifier.\n"
            "Return ONLY a single valid JSON object, no markdown, no extra words.\n"
        )
        prompt = """
Decide if this image depicts an OUTDOOR URBAN BUILT ENVIRONMENT scene.

Return is_urban=true when the scene is mostly man-made: streets/roads, sidewalks, buildings, plazas, bridges, urban waterfront promenades.
Return is_urban=false when the scene is mostly nature/parks/forest/mountain/beach, indoor rooms, portraits/selfies, food closeups, pets, screenshots.

If unsure, set is_urban=false.

Output JSON schema:
{
  "is_urban": <true/false>,
  "confidence": <float 0.0-1.0>,
  "caption": "<one short English sentence describing the scene (<=25 words)>"
}
""".strip()

        payload = {
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
        use_tools = self.use_tools in {"1", "true", "yes", "on", "required"}
        if self.use_tools == "auto":
            use_tools = True
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
            raise RuntimeError(f"HTTP {exc.code} from filter: {body}") from exc

        resp_json = json.loads(raw)
        message = (resp_json.get("choices") or [{}])[0].get("message") or {}
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0].get("function", {}).get("arguments") or ""
            parsed = robust_json_parser(args)
            content = str(args)
        else:
            content = message.get("content") or ""
            parsed = robust_json_parser(content)

        is_urban = _as_bool(parsed.get("is_urban"))
        conf = _as_float(parsed.get("confidence"))
        caption = str(parsed.get("caption") or "")

        if is_urban is None:
            is_urban = False
        if conf is None:
            conf = 0.0
        conf = max(0.0, min(1.0, float(conf)))

        res = UrbanFilterResult(is_urban=bool(is_urban), confidence=conf, caption=caption.strip(), raw=content.strip())
        self._cache[key] = res
        self._save_cache_row(key=key, res=res)
        return res
