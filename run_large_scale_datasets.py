import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from urban_experiment import Sample, UrbanPipeline
from evoshot_env import _load_dotenv


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def reservoir_sample_images(root: Path, k: int, *, recurse: bool = True, seed: int = 42) -> List[Path]:
    rng = random.Random(seed)
    reservoir: List[Path] = []
    seen = 0

    if recurse:
        walker: Iterable[Path] = root.rglob("*")
    else:
        walker = root.glob("*")

    for p in walker:
        if not p.is_file():
            continue
        if not _is_image(p):
            continue

        seen += 1
        if len(reservoir) < k:
            reservoir.append(p)
            continue
        j = rng.randrange(seen)
        if j < k:
            reservoir[j] = p

    return sorted(reservoir)


def _dummy_gt() -> dict:
    return {"safety": 3.0, "vibrancy": 3.0, "cleanliness": 3.0}


@dataclass
class ItemResult:
    dataset: str
    mode: str
    image_path: str
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    retrieved_ids: List[str]
    critique: str
    error: Optional[str] = None


def _categorize_critique(text: str) -> List[str]:
    t = text.lower()
    tags = []
    if any(x in t for x in ["hallucinat", "not grounded", "lacks specific", "lacks specificity", "misses key visual"]):
        tags.append("grounding/hallucination")
    if "vibrancy" in t:
        tags.append("vibrancy")
    if "safety" in t:
        tags.append("safety")
    if "cleanliness" in t:
        tags.append("cleanliness")
    if any(x in t for x in ["park", "nature", "landscape", "not an urban", "non-urban"]):
        tags.append("non_urban/domain_mismatch")
    if not tags and t.strip():
        tags.append("other")
    return tags


def run_dataset(
    *,
    dataset_name: str,
    image_paths: List[Path],
    mode: str,
    post_text: str,
) -> Tuple[List[ItemResult], Dict[str, object]]:
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = mode
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    pipeline = UrbanPipeline()

    results: List[ItemResult] = []
    tags_counter = Counter()
    retrieved_counter = Counter()
    scores: List[float] = []
    pass_scores: List[float] = []

    for idx, image_path in enumerate(image_paths, start=1):
        sample = Sample(
            id=f"{dataset_name}_{idx}",
            image_path=str(image_path),
            post_text=post_text,
            ground_truth_sim=_dummy_gt(),
        )
        trace = pipeline.process_sample(sample)
        teacher_score = trace.get("teacher_score")
        teacher_should_add = trace.get("teacher_should_add")
        retrieved_ids = trace.get("retrieved_ids") or []
        critique = trace.get("teacher_critique") or ""
        error = trace.get("error")

        if isinstance(teacher_score, (int, float)):
            scores.append(float(teacher_score))
            if float(teacher_score) >= 0.8:
                pass_scores.append(float(teacher_score))

        for rid in retrieved_ids:
            retrieved_counter[rid] += 1

        for tag in _categorize_critique(critique):
            tags_counter[tag] += 1

        results.append(
            ItemResult(
                dataset=dataset_name,
                mode=mode,
                image_path=str(image_path),
                teacher_score=float(teacher_score) if isinstance(teacher_score, (int, float)) else None,
                teacher_should_add=bool(teacher_should_add) if teacher_should_add is not None else None,
                retrieved_ids=list(retrieved_ids),
                critique=str(critique),
                error=str(error) if error else None,
            )
        )

    def _avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    summary: Dict[str, object] = {
        "dataset": dataset_name,
        "mode": mode,
        "n": len(image_paths),
        "avg_teacher_score": _avg(scores),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "top_retrieved_ids": retrieved_counter.most_common(10),
        "critique_tags": tags_counter.most_common(),
    }
    return results, summary


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # Real backends
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_STRATEGY", "mmr")
    os.environ.setdefault("EVOSHOT_MMR_LAMBDA", "0.6")

    # Make runs stable by default (override via env if needed).
    # NOTE: Do not force an outdated teacher model here; prefer user's configured default (e.g. Grok 4.1 Fast).
    os.environ.setdefault("EVOSHOT_TEACHER_TEMPERATURE", "0")

    # Ensure student sees image
    os.environ.setdefault("EVOSHOT_LLM_SEND_IMAGE", "1")
    os.environ.setdefault("EVOSHOT_LLM_USE_TOOLS", "auto")
    os.environ.setdefault("EVOSHOT_LLM_TOOL_CHOICE", "required")
    os.environ.setdefault("EVOSHOT_LLM_MAX_TOKENS", "512")

    n1 = int(os.getenv("EVOSHOT_N_DATASET1", "30"))
    n2 = int(os.getenv("EVOSHOT_N_DATASET2", "30"))
    seed = int(os.getenv("EVOSHOT_SAMPLE_SEED", "42"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")

    dataset1_root = Path(r"H:\微博数据 - 副本\images")
    dataset2_roots = [
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\hanover"),
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\aachen"),
    ]

    if not dataset1_root.exists():
        raise FileNotFoundError(f"Dataset1 root not found: {dataset1_root}")
    for r in dataset2_roots:
        if not r.exists():
            raise FileNotFoundError(f"Dataset2 root not found: {r}")

    ds1 = reservoir_sample_images(dataset1_root, k=n1, recurse=True, seed=seed)
    per_root = max(1, (n2 + len(dataset2_roots) - 1) // len(dataset2_roots))
    ds2_parts = []
    for i, r in enumerate(dataset2_roots):
        ds2_parts.extend(reservoir_sample_images(r, k=per_root, recurse=False, seed=seed + i + 1))
    ds2 = ds2_parts[:n2]

    modes = ["text", "caption+text"]

    all_results: List[ItemResult] = []
    summaries: List[Dict[str, object]] = []

    for mode in modes:
        r1, s1 = run_dataset(dataset_name="dataset1_mixed", image_paths=ds1, mode=mode, post_text=post_text)
        all_results.extend(r1)
        summaries.append(s1)

        r2, s2 = run_dataset(dataset_name="dataset2_street", image_paths=ds2, mode=mode, post_text=post_text)
        all_results.extend(r2)
        summaries.append(s2)

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(Path(__file__).resolve().parent / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"large_scale_results_{ts}.json"

    payload = {"summaries": summaries, "results": [asdict(r) for r in all_results]}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"summaries": summaries, "results_path": str(out_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
