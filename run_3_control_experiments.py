import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from evoshot_env import _load_dotenv
from urban_experiment import Sample, UrbanPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def reservoir_sample_images(root: Path, k: int, *, recurse: bool = True, seed: int = 42) -> List[Path]:
    rng = random.Random(seed)
    reservoir: List[Path] = []
    seen = 0

    walker: Iterable[Path] = root.rglob("*") if recurse else root.glob("*")
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


def _categorize_critique(text: str) -> List[str]:
    t = (text or "").lower()
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


@dataclass
class ItemResult:
    experiment: str
    dataset: str
    round: int
    sample_id: str
    image_path: str
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    vault_added_id: Optional[str]
    retrieved_ids: List[str]
    critique: str
    query_caption: Optional[str] = None
    student_scores: Optional[Dict[str, float]] = None
    student_confidence: Optional[float] = None
    error: Optional[str] = None


def _avg(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _summarize_items(items: List[ItemResult]) -> Dict[str, object]:
    scores: List[float] = []
    pass_scores: List[float] = []
    add_flags: List[bool] = []
    tags_counter = Counter()
    retrieved_counter = Counter()

    for r in items:
        if isinstance(r.teacher_score, (int, float)):
            scores.append(float(r.teacher_score))
            if float(r.teacher_score) >= 0.8:
                pass_scores.append(float(r.teacher_score))
        if r.teacher_should_add is not None:
            add_flags.append(bool(r.teacher_should_add))
        for rid in r.retrieved_ids or []:
            retrieved_counter[rid] += 1
        for tag in _categorize_critique(r.critique):
            tags_counter[tag] += 1

    add_rate = (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None

    return {
        "n": len(items),
        "avg_teacher_score": _avg(scores),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "add_rate_should_add": add_rate,
        "critique_tags": tags_counter.most_common(),
        "top_retrieved_ids": retrieved_counter.most_common(10),
    }


def _run_round(
    *,
    pipeline: UrbanPipeline,
    experiment: str,
    dataset: str,
    round_idx: int,
    image_paths: List[Path],
    post_text: str,
) -> List[ItemResult]:
    out: List[ItemResult] = []
    for idx, image_path in enumerate(image_paths, start=1):
        sample_id = f"{dataset}_{idx}"
        sample = Sample(
            id=sample_id,
            image_path=str(image_path),
            post_text=post_text,
            ground_truth_sim=_dummy_gt(),
        )
        trace = pipeline.process_sample(sample)
        out.append(
            ItemResult(
                experiment=experiment,
                dataset=dataset,
                round=round_idx,
                sample_id=sample_id,
                image_path=str(image_path),
                teacher_score=float(trace["teacher_score"]) if isinstance(trace.get("teacher_score"), (int, float)) else None,
                teacher_should_add=bool(trace["teacher_should_add"]) if trace.get("teacher_should_add") is not None else None,
                vault_added_id=trace.get("vault_added_id"),
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                critique=str(trace.get("teacher_critique") or ""),
                query_caption=trace.get("query_caption"),
                student_scores=trace.get("student_scores"),
                student_confidence=float(trace["student_confidence"]) if isinstance(trace.get("student_confidence"), (int, float)) else None,
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def run_experiment(
    *,
    name: str,
    query_mode: str,
    retrieval_strategy: str,
    ds1: List[Path],
    ds2: List[Path],
    post_text: str,
) -> Dict[str, object]:
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = query_mode
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = retrieval_strategy
    os.environ.setdefault("EVOSHOT_RETRIEVAL_K", "2")
    os.environ.setdefault("EVOSHOT_MMR_LAMBDA", "0.6")
    os.environ.setdefault("EVOSHOT_MMR_CANDIDATES", "40")

    # Round 1: allow vault update (learning on)
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    pipeline = UrbanPipeline()

    r1_ds1 = _run_round(pipeline=pipeline, experiment=name, dataset="dataset1_mixed", round_idx=1, image_paths=ds1, post_text=post_text)
    r1_ds2 = _run_round(pipeline=pipeline, experiment=name, dataset="dataset2_street", round_idx=1, image_paths=ds2, post_text=post_text)

    tier_counts = Counter(getattr(ex, "tier", None) for ex in getattr(pipeline.vault, "storage", []) or [])

    # Round 2: freeze vault, re-evaluate same images
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"
    r2_ds1 = _run_round(pipeline=pipeline, experiment=name, dataset="dataset1_mixed", round_idx=2, image_paths=ds1, post_text=post_text)
    r2_ds2 = _run_round(pipeline=pipeline, experiment=name, dataset="dataset2_street", round_idx=2, image_paths=ds2, post_text=post_text)

    all_items = r1_ds1 + r1_ds2 + r2_ds1 + r2_ds2

    def _by(round_idx: int, dataset: str) -> List[ItemResult]:
        return [x for x in all_items if x.round == round_idx and x.dataset == dataset]

    summaries = []
    for round_idx in (1, 2):
        for dataset in ("dataset1_mixed", "dataset2_street"):
            sub = _by(round_idx, dataset)
            s = _summarize_items(sub)
            s.update({"experiment": name, "query_mode": query_mode, "retrieval_strategy": retrieval_strategy, "round": round_idx, "dataset": dataset})
            summaries.append(s)

    return {
        "config": {"experiment": name, "query_mode": query_mode, "retrieval_strategy": retrieval_strategy},
        "vault_tier_counts_after_round1": dict(tier_counts),
        "summaries": summaries,
        "results": [asdict(x) for x in all_items],
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # Real backends
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")

    # Stability / determinism
    os.environ.setdefault("EVOSHOT_LLM_TEMPERATURE", "0")
    os.environ.setdefault("EVOSHOT_TEACHER_TEMPERATURE", "0")
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
    per_root = max(1, n2 // len(dataset2_roots))
    ds2_parts: List[Path] = []
    for i, r in enumerate(dataset2_roots):
        ds2_parts.extend(reservoir_sample_images(r, k=per_root, recurse=False, seed=seed + i + 1))
    ds2 = ds2_parts[:n2]

    experiments = [
        {"name": "E1_original_text_topk", "query_mode": "text", "retrieval_strategy": "topk"},
        {"name": "E2_caption_topk", "query_mode": "caption+text", "retrieval_strategy": "topk"},
        {"name": "E3_caption_mmr", "query_mode": "caption+text", "retrieval_strategy": "mmr"},
    ]

    all_payloads: List[Dict[str, object]] = []
    for cfg in experiments:
        all_payloads.append(
            run_experiment(
                name=cfg["name"],
                query_mode=cfg["query_mode"],
                retrieval_strategy=cfg["retrieval_strategy"],
                ds1=ds1,
                ds2=ds2,
                post_text=post_text,
            )
        )

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"control_3_experiments_{ts}.json"
    out_json.write_text(json.dumps({"experiments": all_payloads}, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print compact summaries
    compact = []
    for payload in all_payloads:
        for s in payload.get("summaries") or []:
            compact.append(s)
    print(json.dumps({"summaries": compact, "results_path": str(out_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

