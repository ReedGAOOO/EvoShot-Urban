import json
import os
import re
import urllib.request
from typing import Any, Dict, Iterable, Optional, Type


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
        timeout_s: float = 120.0,
    ):
        self._output_model = output_model
        self._score_dims = list(score_dims)
        self._url = (base_url or os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip()
        self._model = (model or os.getenv("EVOSHOT_LLM_MODEL") or "qwen3-vl-30b-a3b-thinking").strip()
        self._temperature = float(os.getenv("EVOSHOT_LLM_TEMPERATURE", str(temperature)))
        self._timeout_s = float(os.getenv("EVOSHOT_LLM_TIMEOUT_S", str(timeout_s)))

    def _build_prompt(self, *, image_path: str, post_text: str, shots_text: str) -> str:
        dims = ", ".join(self._score_dims)
        return (
            "You are an urban perception scoring model.\n"
            f"Return ONLY a single JSON object with keys: rationale (string), scores (object), model_confidence (number 0-1).\n"
            f"'scores' must contain numeric values (floats) for: {dims}. Range 1-5.\n"
            "No markdown, no backticks, no extra text.\n\n"
            f"Few-shot examples:\n{shots_text}\n"
            f"Now score this input:\n"
            f"Image: {image_path}\n"
            f"Text: {post_text}\n"
        )

    def _call_chat_completions(self, prompt: str) -> Dict[str, Any]:
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "stream": False,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)

    def _extract_content(self, response_json: Dict[str, Any]) -> str:
        try:
            return response_json["choices"][0]["message"]["content"]
        except Exception as exc:
            raise ValueError(f"Unexpected chat/completions response shape: {response_json}") from exc

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

    def predict(self, sample: Any, shots: Any) -> Any:
        shots_text = "".join(ex.to_prompt_str() for ex in shots) if shots else ""
        prompt = self._build_prompt(image_path=sample.image_path, post_text=sample.post_text, shots_text=shots_text)

        response_json = self._call_chat_completions(prompt)
        content = self._extract_content(response_json)
        parsed = robust_json_parser(content)
        normalized = self._coerce_output(parsed)
        return self._output_model(**normalized)
