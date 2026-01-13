import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _bool_env(name: str, default: bool = False) -> bool:
    val = (os.getenv(name) or "").strip().lower()
    if not val:
        return default
    return val in {"1", "true", "yes", "on"}

def _load_dotenv(dotenv_path: Path, override: bool = False) -> None:
    try:
        raw = dotenv_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not override and os.getenv(key) is not None:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ[key] = value


@dataclass(frozen=True)
class EvoShotEnv:
    project_root: Path

    # Backends
    student_backend: str
    embed_backend: str
    teacher_backend: str

    # Student (OpenAI-compatible) endpoint
    student_llm_url: str
    student_llm_model: str
    student_llm_send_image: str
    student_llm_use_tools: str
    student_llm_tool_choice: str

    # Embedding endpoint (as in embedding_test.py)
    embed_api_url: str
    embed_instruction: str
    embed_timeout_s: float
    embed_image_path_style: str

    # Teacher (OpenAI)
    teacher_model: str
    teacher_caption_model: str
    openai_api_key: Optional[str]
    openrouter_api_key: Optional[str]
    teacher_api_url: str
    openrouter_site_url: Optional[str]
    openrouter_site_name: Optional[str]

    # Local storage
    data_dir: Path
    images_dir: Path
    texts_dir: Path
    vault_dir: Path
    logs_dir: Path

    @staticmethod
    def load(project_root: Optional[Path] = None) -> "EvoShotEnv":
        root = (project_root or Path(__file__).resolve().parent).resolve()
        _load_dotenv(root / ".env", override=False)
        data_dir = Path(os.getenv("EVOSHOT_DATA_DIR", str(root / "data"))).resolve()
        images_dir = Path(os.getenv("EVOSHOT_IMAGES_DIR", str(data_dir / "images"))).resolve()
        texts_dir = Path(os.getenv("EVOSHOT_TEXTS_DIR", str(data_dir / "texts"))).resolve()
        vault_dir = Path(os.getenv("EVOSHOT_VAULT_DIR", str(data_dir / "vault"))).resolve()
        logs_dir = Path(os.getenv("EVOSHOT_LOGS_DIR", str(data_dir / "logs"))).resolve()

        openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or None
        openrouter_api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip() or None

        teacher_model_default = "gpt-4o-2024-08-06"
        if openrouter_api_key:
            teacher_model_default = "x-ai/grok-4.1-fast"

        teacher_api_url = (os.getenv("EVOSHOT_TEACHER_API_URL") or "").strip()
        if not teacher_api_url:
            teacher_api_url = (
                "https://openrouter.ai/api/v1/chat/completions"
                if openrouter_api_key
                else "https://api.openai.com/v1/chat/completions"
            )

        return EvoShotEnv(
            project_root=root,
            student_backend=(os.getenv("EVOSHOT_STUDENT_BACKEND") or "mock").strip(),
            embed_backend=(os.getenv("EVOSHOT_EMBED_BACKEND") or "mock").strip(),
            teacher_backend=(os.getenv("EVOSHOT_TEACHER_BACKEND") or "mock").strip(),
            student_llm_url=(os.getenv("EVOSHOT_LLM_URL") or "http://100.76.172.42:1234/v1/chat/completions").strip(),
            student_llm_model=(os.getenv("EVOSHOT_LLM_MODEL") or "qwen3-vl-30b-a3b-thinking").strip(),
            student_llm_send_image=(os.getenv("EVOSHOT_LLM_SEND_IMAGE") or "auto").strip(),
            student_llm_use_tools=(os.getenv("EVOSHOT_LLM_USE_TOOLS") or "auto").strip(),
            student_llm_tool_choice=(os.getenv("EVOSHOT_LLM_TOOL_CHOICE") or "required").strip(),
            embed_api_url=(os.getenv("EVOSHOT_EMBED_API") or "http://localhost:8000/embed").strip(),
            embed_instruction=(
                os.getenv("EVOSHOT_EMBED_INSTRUCTION") or "Retrieve images or text relevant to the user query."
            ),
            embed_timeout_s=float(os.getenv("EVOSHOT_EMBED_TIMEOUT_S", "120")),
            embed_image_path_style=(os.getenv("EVOSHOT_EMBED_IMAGE_PATH_STYLE") or "windows").strip(),
            teacher_model=(os.getenv("EVOSHOT_TEACHER_MODEL") or teacher_model_default).strip(),
            teacher_caption_model=(os.getenv("EVOSHOT_TEACHER_CAPTION_MODEL") or teacher_model_default).strip(),
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            teacher_api_url=teacher_api_url,
            openrouter_site_url=(os.getenv("OPENROUTER_SITE_URL") or "").strip() or None,
            openrouter_site_name=(os.getenv("OPENROUTER_SITE_NAME") or os.getenv("OPENROUTER_APP_NAME") or "").strip()
            or None,
            data_dir=data_dir,
            images_dir=images_dir,
            texts_dir=texts_dir,
            vault_dir=vault_dir,
            logs_dir=logs_dir,
        )

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.images_dir, self.texts_dir, self.vault_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict:
        return {
            "student_backend": self.student_backend,
            "embed_backend": self.embed_backend,
            "teacher_backend": self.teacher_backend,
            "student_llm_url": self.student_llm_url,
            "student_llm_model": self.student_llm_model,
            "student_llm_send_image": self.student_llm_send_image,
            "student_llm_use_tools": self.student_llm_use_tools,
            "student_llm_tool_choice": self.student_llm_tool_choice,
            "embed_api_url": self.embed_api_url,
            "embed_instruction": self.embed_instruction,
            "embed_timeout_s": self.embed_timeout_s,
            "embed_image_path_style": self.embed_image_path_style,
            "teacher_model": self.teacher_model,
            "teacher_caption_model": self.teacher_caption_model,
            "openai_api_key_present": bool(self.openai_api_key),
            "openrouter_api_key_present": bool(self.openrouter_api_key),
            "teacher_api_url": self.teacher_api_url,
            "openrouter_site_url": self.openrouter_site_url,
            "openrouter_site_name": self.openrouter_site_name,
            "data_dir": str(self.data_dir),
            "images_dir": str(self.images_dir),
            "texts_dir": str(self.texts_dir),
            "vault_dir": str(self.vault_dir),
            "logs_dir": str(self.logs_dir),
        }
