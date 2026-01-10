import json
import os
import re
import base64
import mimetypes
import urllib.request
from typing import Any, Dict, Iterable, Optional, Type, Tuple


def _extract_balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    depth = 0
    start: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : idx + 1])
                start = None
    return objects


def _looks_like_url(path_or_url: str) -> bool:
    p = path_or_url.strip().lower()
    return p.startswith("http://") or p.startswith("https://") or p.startswith("data:image/")


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}", mime


def robust_json_parser(raw_text: str) -> dict:
    """
    清洗大模型输出的脏 JSON
    """
    # 1. 尝试直接解析
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    # 2. 提取 markdown 代码块 ```json ... ```
    match = re.search(r"```json\\s*(\\{.*?\\})\\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 2.1 兼容未标注 json 的代码块 ``` ... ```
    match = re.search(r"```\\s*(\\{.*?\\})\\s*```", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. 提取最外层大括号 {...}
    for candidate in reversed(_extract_balanced_json_objects(raw_text)):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # 4. 如果是单引号，尝试替换为双引号（常见错误）
    try:
        fixed_text = raw_text.replace("'", '"')
        return json.loads(fixed_text)
    except Exception as exc:
        snippet = raw_text[:200].replace("\n", "\\n")
        raise ValueError(f"Failed to parse JSON from: {snippet}...") from exc


class HTTPStudentModel:
    """
    通过 OpenAI 兼容接口调用真实 LLM，并用 robust_json_parser 解析输出。

    约定：模型输出应尽量是 JSON（允许被 <think> / 前缀 / 代码块包裹）。
    """

    def __init__(
        self,
        *,
        output_model: Type[Any],
        score_dims: Iterable[str],
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout_s: float = 120.0,
    ):
        self._output_model = output_model
        self._score_dims = list(score_dims)
        self._url = (base_url or os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip()
        self._model = (model or os.getenv("EVOSHOT_LLM_MODEL") or "qwen3-vl-30b-a3b-thinking").strip()
        self._temperature = float(os.getenv("EVOSHOT_LLM_TEMPERATURE", str(temperature)))
        self._max_tokens = int(os.getenv("EVOSHOT_LLM_MAX_TOKENS", str(max_tokens)))
        self._timeout_s = float(os.getenv("EVOSHOT_LLM_TIMEOUT_S", str(timeout_s)))
        self._send_image = (os.getenv("EVOSHOT_LLM_SEND_IMAGE") or "auto").strip().lower()
        self._debug = (os.getenv("EVOSHOT_LLM_DEBUG") or "").strip().lower() in {"1", "true", "yes", "on"}
        self._json_retry = (os.getenv("EVOSHOT_LLM_JSON_RETRY") or "1").strip().lower() in {"1", "true", "yes", "on"}
        self._response_format = (os.getenv("EVOSHOT_LLM_RESPONSE_FORMAT") or "").strip().lower()
        self._repair_model = (os.getenv("EVOSHOT_LLM_REPAIR_MODEL") or self._model).strip()
        self._use_tools = (os.getenv("EVOSHOT_LLM_USE_TOOLS") or "auto").strip().lower()
        self._tool_choice = (os.getenv("EVOSHOT_LLM_TOOL_CHOICE") or "required").strip().lower()

    def _system_prompt(self) -> str:
        dims = ", ".join(self._score_dims)
        return (
            "You are an intelligent urban environment analyzer.\n"
            f"You MUST extract scores from BOTH the image and the text: {dims}.\n"
            "Each score range: 1.0 to 5.0.\n\n"
            "RESPONSE FORMAT:\n"
            "Strictly valid JSON only. No markdown, no backticks, no conversational filler, no <think>.\n"
            "Example:\n"
            '{"rationale":"...","scores":{"safety":3.5,"vibrancy":4.0,"cleanliness":2.5},"model_confidence":0.8}'
        )

    def _repair_system_prompt(self) -> str:
        return (
            "You are a strict JSON reformatter.\n"
            "Output ONLY a single valid JSON object and nothing else.\n"
            "No markdown, no backticks, no extra words.\n"
        )

    def _tools_spec(self) -> list[Dict[str, Any]]:
        props = {k: {"type": "number"} for k in self._score_dims}
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_scores",
                    "description": "Submit urban perception scores and rationale.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "scores": {
                                "type": "object",
                                "properties": props,
                                "required": list(self._score_dims),
                            },
                            "model_confidence": {"type": "number"},
                        },
                        "required": ["rationale", "scores", "model_confidence"],
                    },
                },
            }
        ]

    def _build_user_text(self, *, image_path: str, post_text: str, shots_text: str) -> str:
        dims = ", ".join(self._score_dims)
        return (
            "Reference Examples:\n"
            f"{shots_text}\n"
            "Now analyze this NEW target:\n"
            f"Text: {post_text}\n"
            "[Image Attached]\n"
            f"(Reminder: scores keys = {dims}; return ONLY JSON.)\n"
        )

    def _build_payload(self, *, user_text: str, image_path: str, use_tools: bool) -> Dict[str, Any]:
        messages: list[Dict[str, Any]] = [{"role": "system", "content": self._system_prompt()}]

        include_image = False
        if self._send_image in {"1", "true", "yes", "on"}:
            include_image = True
        elif self._send_image in {"0", "false", "no", "off"}:
            include_image = False
        else:
            include_image = os.path.exists(image_path)

        if include_image and _looks_like_url(image_path):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_path}},
                    ],
                }
            )
        elif include_image and os.path.exists(image_path):
            data_url, _mime = _encode_image_data_url(image_path)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
        elif include_image and not os.path.exists(image_path) and not _looks_like_url(image_path):
            raise FileNotFoundError(
                f"EVOSHOT_LLM_SEND_IMAGE requested but image file not found: {image_path}"
            )
        else:
            messages.append({"role": "user", "content": user_text.replace("[Image Attached]", f"Image: {image_path}")})

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if use_tools:
            payload["tools"] = self._tools_spec()
            payload["tool_choice"] = self._tool_choice
        if self._response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _call_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._debug:
            img_mode = self._send_image
            print(f"[DEBUG] POST {self._url} model={payload.get('model')} send_image={img_mode} max_tokens={payload.get('max_tokens')}")
        body = payload
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from LLM: {body}") from exc

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        try:
            return response_json["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ValueError(f"Unexpected chat/completions response shape: {response_json}") from exc

    def _extract_tool_arguments(self, response_json: Dict[str, Any]) -> Optional[str]:
        try:
            tool_calls = response_json["choices"][0]["message"].get("tool_calls") or []
            if not tool_calls:
                return None
            return tool_calls[0]["function"]["arguments"]
        except Exception:
            return None

    def _coerce_output(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        rationale = parsed.get("rationale") or parsed.get("reason") or ""

        scores: Dict[str, Any]
        if isinstance(parsed.get("scores"), dict):
            scores = dict(parsed["scores"])
        else:
            scores = {k: parsed.get(k) for k in self._score_dims if k in parsed}

        fixed_scores: Dict[str, float] = {}
        for dim in self._score_dims:
            val = scores.get(dim, 1.0)
            if isinstance(val, str):
                try:
                    val = float(val)
                except Exception:
                    val = 1.0
            try:
                val_f = float(val)
            except Exception:
                val_f = 1.0
            fixed_scores[dim] = max(1.0, min(5.0, round(val_f, 2)))

        conf = parsed.get("model_confidence", parsed.get("confidence", 0.5))
        try:
            conf_f = float(conf)
        except Exception:
            conf_f = 0.5
        conf_f = max(0.0, min(1.0, conf_f))

        return {"rationale": str(rationale), "scores": fixed_scores, "model_confidence": conf_f}

    def _repair_to_json(self, *, raw_content: str) -> Dict[str, Any]:
        schema_hint = (
            "Return a single JSON object with keys:\n"
            "- rationale: string\n"
            "- scores: object with numeric keys safety, vibrancy, cleanliness (range 1.0-5.0)\n"
            "- model_confidence: number in [0,1]\n"
        )
        repair_payload: Dict[str, Any] = {
            "model": self._repair_model,
            "messages": [
                {"role": "system", "content": self._repair_system_prompt()},
                {
                    "role": "user",
                    "content": (
                        f"{schema_hint}\n"
                        "Convert the following text into STRICT JSON ONLY.\n"
                        "Your response MUST start with '{' and end with '}'.\n\n"
                        f"{raw_content}"
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if self._response_format == "json_object":
            repair_payload["response_format"] = {"type": "json_object"}
        response_json = self._call_chat_completions(repair_payload)
        tool_args = self._extract_tool_arguments(response_json)
        content = tool_args if tool_args is not None else self._extract_content(response_json)
        if self._debug:
            print(f"\n[DEBUG RAW LLM REPAIR RESPONSE]:\n{content}\n")
        return robust_json_parser(content)

    def predict(self, sample: Any, shots: Any) -> Any:
        shots_text = "".join(ex.to_prompt_str() for ex in shots) if shots else ""
        user_text = self._build_user_text(image_path=sample.image_path, post_text=sample.post_text, shots_text=shots_text)
        use_tools = self._use_tools in {"1", "true", "yes", "on", "required"}
        if self._use_tools == "auto":
            use_tools = True
        payload = self._build_payload(user_text=user_text, image_path=sample.image_path, use_tools=use_tools)

        try:
            response_json = self._call_chat_completions(payload)
        except RuntimeError:
            if self._use_tools != "auto" or not use_tools:
                raise
            payload = self._build_payload(user_text=user_text, image_path=sample.image_path, use_tools=False)
            response_json = self._call_chat_completions(payload)

        tool_args = self._extract_tool_arguments(response_json)
        content = tool_args if tool_args is not None else self._extract_content(response_json)
        if self._debug:
            print(f"\n[DEBUG RAW LLM RESPONSE]:\n{content}\n")
        try:
            parsed = robust_json_parser(content)
        except Exception:
            if not self._json_retry:
                raise
            parsed = self._repair_to_json(raw_content=content)
        normalized = self._coerce_output(parsed)
        return self._output_model(**normalized)
