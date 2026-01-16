import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .student import robust_json_parser


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
    """
    Encode an image as a data URL for vision captioning.

    To keep caption calls fast, we downsize images by default:
      - EVOSHOT_CAPTION_IMAGE_MAX_SIDE (default 640)
      - EVOSHOT_CAPTION_IMAGE_JPEG_QUALITY (default 85)
    """
    max_side = int(os.getenv("EVOSHOT_CAPTION_IMAGE_MAX_SIDE", "640"))
    quality = int(os.getenv("EVOSHOT_CAPTION_IMAGE_JPEG_QUALITY", "85"))

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


def _strip_think(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()

    # 1) Remove well-formed <think>...</think> blocks.
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2) Handle unclosed <think> (common for "thinking" models): drop the tag itself.
    cleaned = re.sub(r"<think>\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # 3) Heuristic: strip common preamble and keep the actual observation part if present.
    m = re.search(r"first,?\s*observe\s*the\s*image\s*:?", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = cleaned[m.end() :].strip()

    # 4) Normalize whitespace.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class LocalVisionCaptioner:
    """
    Caption an image via the same OpenAI-compatible endpoint as the Student.
    Used only to build a better retrieval query embedding (cheap, local).
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = 60.0,
    ):
        self.url = (base_url or os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip()
        self.model = (model or os.getenv("EVOSHOT_CAPTION_MODEL") or os.getenv("EVOSHOT_LLM_MODEL") or "qwen3-vl-30b-a3b-thinking").strip()
        self.timeout_s = float(os.getenv("EVOSHOT_CAPTION_TIMEOUT_S", str(timeout_s)))
        self.max_tokens = int(os.getenv("EVOSHOT_CAPTION_MAX_TOKENS", "120"))

        root = Path(__file__).resolve().parent.parent
        data_dir = Path(os.getenv("EVOSHOT_DATA_DIR", str(root / "data"))).resolve()
        default_cache = data_dir / "cache" / "caption_cache.jsonl"
        self.cache_path = Path(os.getenv("EVOSHOT_CAPTION_CACHE_PATH", str(default_cache))).resolve()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._cache: dict[str, str] = {}
        self._load_cache()

        # Retrieval-oriented structured summaries (more stable for embedding than free-form captions).
        # NOTE: bumped default version to invalidate earlier cached "unknown-only" summaries.
        self.summary_version = (os.getenv("EVOSHOT_RETRIEVAL_SUMMARY_VERSION") or "v2").strip()
        default_summary_cache = data_dir / "cache" / "retrieval_summary_cache.jsonl"
        self.summary_cache_path = Path(
            os.getenv("EVOSHOT_RETRIEVAL_SUMMARY_CACHE_PATH", str(default_summary_cache))
        ).resolve()
        self.summary_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_cache: dict[str, str] = {}
        self._load_summary_cache()

    def _cache_key(self, image_path: str) -> str:
        try:
            p = Path(image_path)
            if p.exists():
                return f"{self.model}::{p.resolve()}"
        except Exception:
            pass
        return f"{self.model}::{image_path}"

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
                caption = str(obj.get("caption") or "")
                if key and caption:
                    self._cache[key] = caption
        except Exception:
            return

    def _summary_cache_key(self, image_path: str) -> str:
        try:
            p = Path(image_path)
            if p.exists():
                return f"{self.summary_version}::{self.model}::{p.resolve()}"
        except Exception:
            pass
        return f"{self.summary_version}::{self.model}::{image_path}"

    def _load_summary_cache(self) -> None:
        if not self.summary_cache_path.exists():
            return
        try:
            for line in self.summary_cache_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = str(obj.get("key") or "")
                summary = str(obj.get("summary") or "")
                if key and summary:
                    self._summary_cache[key] = summary
        except Exception:
            return

    def _save_summary_cache_row(self, *, key: str, summary: str, raw: str) -> None:
        try:
            with self.summary_cache_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"key": key, "summary": summary, "raw": raw, "model": self.model, "version": self.summary_version},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception:
            return

    @staticmethod
    def _canonicalize_summary(data: Dict[str, Any]) -> str:
        def pick_str(key: str, default: str = "unknown") -> str:
            v = data.get(key, default)
            s = str(v).strip().lower()
            return s or default

        scene = pick_str("scene")
        lighting = pick_str("lighting")
        crowd = pick_str("crowd")
        traffic = pick_str("traffic")
        trash = pick_str("trash")
        graffiti = pick_str("graffiti")
        sidewalk = pick_str("sidewalk")

        objects: List[str] = []
        raw_obj = data.get("objects")
        if isinstance(raw_obj, list):
            for x in raw_obj[:8]:
                s = str(x).strip().lower()
                if s:
                    objects.append(s)

        parts = [
            f"scene={scene}",
            f"lighting={lighting}",
            f"crowd={crowd}",
            f"traffic={traffic}",
            f"trash={trash}",
            f"graffiti={graffiti}",
            f"sidewalk={sidewalk}",
        ]
        if objects:
            parts.append("objects=" + "|".join(objects))
        return "; ".join(parts)

    @staticmethod
    def _summary_tools_spec() -> List[Dict[str, Any]]:
        enums = {
            "scene": ["street", "alley", "intersection", "plaza", "parking", "indoor", "other", "unknown"],
            "lighting": ["day_bright", "night_well_lit", "night_dim", "indoor", "unknown"],
            "crowd": ["none", "low", "medium", "high", "unknown"],
            "traffic": ["none", "low", "medium", "high", "unknown"],
            "trash": ["none", "low", "medium", "high", "unknown"],
            "graffiti": ["none", "low", "medium", "high", "unknown"],
            "sidewalk": ["none", "good", "cracked", "broken", "dirty", "unknown"],
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_retrieval_summary",
                    "description": "Submit a structured attribute summary for retrieval embedding.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scene": {"type": "string", "enum": enums["scene"]},
                            "lighting": {"type": "string", "enum": enums["lighting"]},
                            "crowd": {"type": "string", "enum": enums["crowd"]},
                            "traffic": {"type": "string", "enum": enums["traffic"]},
                            "trash": {"type": "string", "enum": enums["trash"]},
                            "graffiti": {"type": "string", "enum": enums["graffiti"]},
                            "sidewalk": {"type": "string", "enum": enums["sidewalk"]},
                            "objects": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["scene", "lighting", "crowd", "traffic", "trash", "graffiti", "sidewalk", "objects"],
                    },
                },
            }
        ]

    @staticmethod
    def _extract_tool_arguments(resp_json: dict) -> Optional[str]:
        try:
            msg = (resp_json.get("choices") or [])[0].get("message") or {}
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return None
            return tool_calls[0]["function"]["arguments"]
        except Exception:
            return None

    def structured_summary(self, image_path: str) -> str:
        """
        Return a structured attribute summary for retrieval embedding.

        This is used ONLY for building retrieval queries; do NOT store it as the vault's `image_desc`.
        """
        key = self._summary_cache_key(image_path)
        cached = self._summary_cache.get(key)
        if cached is not None:
            return cached

        data_url, _mime = _encode_image_data_url(image_path)
        prompt = (
            "Extract a compact attribute summary of this scene for similarity search.\n"
            "Use the function tool to submit the fields.\n\n"
            "Guidelines:\n"
            "- Choose the closest label from each enum.\n"
            "- objects: up to 8 short nouns (e.g., \"car\", \"shop_sign\", \"trash_bin\").\n"
        )
        use_tools = (os.getenv("EVOSHOT_RETRIEVAL_SUMMARY_USE_TOOLS") or "auto").strip().lower()
        tool_choice = (os.getenv("EVOSHOT_RETRIEVAL_SUMMARY_TOOL_CHOICE") or "required").strip().lower()
        enable_tools = use_tools in {"1", "true", "yes", "on", "required"}
        if use_tools == "auto":
            enable_tools = True

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract attributes for similarity search.\n"
                        "Be grounded in the image. Avoid hallucinations.\n"
                        "Submit results via the provided function tool.\n"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.0,
            "max_tokens": 220,
            "stream": False,
        }
        if enable_tools:
            payload["tools"] = self._summary_tools_spec()
            payload["tool_choice"] = tool_choice

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from retrieval summary: {body}") from exc

        resp_json = json.loads(raw)
        tool_args = self._extract_tool_arguments(resp_json)
        content = tool_args if tool_args is not None else (resp_json["choices"][0]["message"].get("content") or "").strip()
        cleaned = _strip_think(content or "")
        try:
            parsed = robust_json_parser(cleaned)
            summary = self._canonicalize_summary(parsed if isinstance(parsed, dict) else {})
        except Exception:
            summary = ""

        if summary:
            self._summary_cache[key] = summary
            self._save_summary_cache_row(key=key, summary=summary, raw=cleaned)

        return summary

    def caption(self, image_path: str) -> str:
        key = self._cache_key(image_path)
        cached = self._cache.get(key)
        if cached is not None:
            cleaned = _strip_think(cached)
            if cleaned and cleaned != cached:
                self._cache[key] = cleaned
            return cleaned

        data_url, _mime = _encode_image_data_url(image_path)
        prompt = (
            "Describe this urban street scene in 40 words max. "
            "Mention lighting, crowd/traffic level, cleanliness, and notable objects. "
            "Return ONE sentence only. No markdown."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return a single sentence caption. No <think>."},
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
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from captioner: {body}") from exc
        resp_json = json.loads(raw)
        content = resp_json["choices"][0]["message"].get("content") or ""
        caption = _strip_think(content)
        if caption:
            self._cache[key] = caption
            try:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps({"key": key, "caption": caption, "model": self.model}, ensure_ascii=False) + "\n"
                    )
            except Exception:
                pass
        return caption
