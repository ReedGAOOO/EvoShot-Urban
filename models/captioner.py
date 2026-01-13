import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple


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
