import base64
import json
import logging
import mimetypes
import os
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
        self.model = (os.getenv("EVOSHOT_TEACHER_MODEL") or model_name).strip()
        self.caption_model = (os.getenv("EVOSHOT_TEACHER_CAPTION_MODEL") or caption_model_name or self.model).strip()
        self.temperature = float(os.getenv("EVOSHOT_TEACHER_TEMPERATURE", "0.2"))
        self.timeout_s = float(os.getenv("EVOSHOT_TEACHER_TIMEOUT_S", "120"))

        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:
            raise ImportError(
                "Missing dependency `openai`. Install it (e.g., `pip install openai`) or keep EVOSHOT_TEACHER_BACKEND=mock."
            ) from exc

        self.client = OpenAI()

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
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self._judge_messages(sample, student_out),
                response_format={"type": "json_object"},
                temperature=self.temperature,
                timeout=self.timeout_s,
            )
            raw = resp.choices[0].message.content or ""
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
            resp = self.client.chat.completions.create(
                model=self.caption_model,
                messages=[
                    {"role": "system", "content": "Return a single short sentence. No markdown."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0.2,
                timeout=self.timeout_s,
            )
            caption = (resp.choices[0].message.content or "").strip()
            return caption or "Urban scene description unavailable."
        except Exception as exc:
            logger.error(f"Teacher Caption Failed: {exc}")
            return "Urban scene description unavailable."

