import base64
import json
import logging
import mimetypes
import os
import urllib.request
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from .student import robust_json_parser

logger = logging.getLogger("UrbanExp")


def _encode_image_data_url(image_path: str) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}", mime


class TeacherFeedback(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)  # 0.0 - 1.0 (Reward)
    critique: str
    better_rationale: str
    better_scores: Dict[str, float]
    should_add_to_vault: bool


class OpenAITeacher:
    def __init__(
        self,
        model_name: str = "gpt-4o-2024-08-06",
        caption_model_name: Optional[str] = None,
    ):
        openrouter_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        openai_key = (os.getenv("OPENAI_API_KEY") or "").strip()

        self._provider = "openrouter" if openrouter_key else "openai"
        self._api_key = openrouter_key or openai_key
        if not self._api_key:
            raise ValueError(
                "Missing API key for real teacher. Set OPENROUTER_API_KEY (recommended) or OPENAI_API_KEY, "
                "or keep EVOSHOT_TEACHER_BACKEND=mock."
            )

        if self._provider == "openrouter":
            default_model = "x-ai/grok-4.1-fast"
        else:
            default_model = model_name

        self.model = (os.getenv("EVOSHOT_TEACHER_MODEL") or default_model).strip()
        self.caption_model = (os.getenv("EVOSHOT_TEACHER_CAPTION_MODEL") or caption_model_name or self.model).strip()
        self.temperature = float(os.getenv("EVOSHOT_TEACHER_TEMPERATURE", "0.2"))
        self.timeout_s = float(os.getenv("EVOSHOT_TEACHER_TIMEOUT_S", "120"))

        self._api_url = (os.getenv("EVOSHOT_TEACHER_API_URL") or "").strip()
        if not self._api_url:
            if self._provider == "openrouter":
                self._api_url = "https://openrouter.ai/api/v1/chat/completions"
            else:
                self._api_url = "https://api.openai.com/v1/chat/completions"

        self._openrouter_site_url = (os.getenv("OPENROUTER_SITE_URL") or "").strip()
        self._openrouter_title = (os.getenv("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_APP_NAME") or "").strip()
        logger.info(f"Teacher provider={self._provider} model={self.model} api_url={self._api_url}")

    def _call_chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._provider == "openrouter":
            if self._openrouter_site_url:
                headers["HTTP-Referer"] = self._openrouter_site_url
            if self._openrouter_title:
                headers["X-Title"] = self._openrouter_title

        req = urllib.request.Request(
            self._api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Teacher HTTP call failed: {exc}") from exc

        try:
            data = json.loads(raw)
        except Exception as exc:
            snippet = raw[:200].replace("\n", "\\n")
            raise RuntimeError(f"Teacher returned non-JSON: {snippet}...") from exc

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Teacher API error: {data['error']}")
        return data

    @staticmethod
    def _extract_content(response_json: Dict[str, Any]) -> str:
        try:
            choices = response_json.get("choices") or []
            message = (choices[0] or {}).get("message") or {}
            return (message.get("content") or "").strip()
        except Exception:
            return ""

    def _judge_messages(self, sample: Any, student_out: Any) -> list[dict]:
        image_path = getattr(sample, "image_path", None)
        if not image_path:
            raise ValueError("sample.image_path is required for vision teacher grading.")
        data_url, _mime = _encode_image_data_url(str(image_path))

        user_content = [
            {
                "type": "text",
                "text": f"""
Task: Evaluate the Student's analysis of an urban scene. Be strict and grounded in the image.

[User Post Context]: "{getattr(sample, "post_text", "")}"

[Student Output]:
Scores: {json.dumps(getattr(student_out, "scores", {}), ensure_ascii=False)}
Rationale: "{getattr(student_out, "rationale", "")}"

Evaluation Criteria:
1. Grounding: Did the student mention specific visual details visible in the image? (Crucial)
2. Logic: Do the scores match the rationale?
3. Nuance: Did the student miss obvious safety/cleanliness issues?

Output JSON:
{{
  "score": <float 0.0-1.0, be harsh>,
  "critique": "<short criticism>",
  "better_rationale": "<write a perfect rationale grounded in the image>",
  "better_scores": {{"safety": <float 1-5>, "vibrancy": <float 1-5>, "cleanliness": <float 1-5>}},
  "should_add_to_vault": <bool, true only if student failed (score < 0.8)>
}}
""".strip(),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]

        return [
            {
                "role": "system",
                "content": "You are a strict Senior Urban Planner. You grade Junior AI outputs and punish hallucinations.",
            },
            {"role": "user", "content": user_content},
        ]

    def judge(self, sample: Any, student_out: Any) -> TeacherFeedback:
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": self._judge_messages(sample, student_out),
                "temperature": self.temperature,
                "stream": False,
            }
            response_json = self._call_chat_completions(payload)
            raw = self._extract_content(response_json)
            try:
                data = json.loads(raw)
            except Exception:
                data = robust_json_parser(raw)

            if "should_add_to_vault" not in data and "score" in data:
                try:
                    data["should_add_to_vault"] = float(data["score"]) < 0.8
                except Exception:
                    data["should_add_to_vault"] = False

            if isinstance(data.get("better_scores"), dict):
                fixed: Dict[str, float] = {}
                for k in ("safety", "vibrancy", "cleanliness"):
                    v = data["better_scores"].get(k, 1.0)
                    try:
                        fixed[k] = float(v)
                    except Exception:
                        fixed[k] = 1.0
                data["better_scores"] = fixed

            return TeacherFeedback(**data)
        except Exception as exc:
            logger.error(f"Teacher Judge Failed: {exc}")
            return TeacherFeedback(
                score=0.0,
                critique="Teacher Error",
                better_rationale="",
                better_scores={},
                should_add_to_vault=False,
            )

    def generate_caption(self, sample: Any) -> str:
        image_path = getattr(sample, "image_path", None)
        if not image_path:
            return "Urban scene description unavailable."
        try:
            data_url, _mime = _encode_image_data_url(str(image_path))
            prompt = "Describe this urban scene in 30 words. Focus on safety, vibrancy, and cleanliness features."
            payload: Dict[str, Any] = {
                "model": self.caption_model,
                "messages": [
                    {"role": "system", "content": "Return a single short sentence. No markdown."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                "temperature": 0.2,
                "stream": False,
            }
            response_json = self._call_chat_completions(payload)
            caption = self._extract_content(response_json)
            return caption or "Urban scene description unavailable."
        except Exception as exc:
            logger.error(f"Teacher Caption Failed: {exc}")
            return "Urban scene description unavailable."
