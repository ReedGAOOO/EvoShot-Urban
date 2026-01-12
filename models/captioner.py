import base64
import json
import mimetypes
import os
import re
import urllib.error
import urllib.request
from typing import Optional, Tuple


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
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
    cleaned = re.sub(r"<think>.*?</think>\\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # 2) Handle unclosed <think> (common for "thinking" models): drop the tag itself.
    cleaned = re.sub(r"<think>\\s*", "", cleaned, flags=re.IGNORECASE).strip()

    # 3) Heuristic: strip common preamble and keep the actual observation part if present.
    m = re.search(r"first,?\\s*observe\\s*the\\s*image\\s*:?", cleaned, flags=re.IGNORECASE)
    if m:
        cleaned = cleaned[m.end() :].strip()

    # 4) Normalize whitespace.
    cleaned = re.sub(r"\\s+", " ", cleaned).strip()
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

    def caption(self, image_path: str) -> str:
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
        return _strip_think(content)
