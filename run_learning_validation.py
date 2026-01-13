import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from evoshot_env import _load_dotenv
from urban_experiment import Sample, UrbanPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def reservoir_sample_images(root: Path, k: int, *, recurse: bool, seed: int) -> List[Path]:
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


def sample_train_test_cityscapes(
    *,
    roots: List[Path],
    train_n: int,
    test_n: int,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    total = train_n + test_n
    if total <= 0:
        return [], []

    per_root_total = max(1, (total + len(roots) - 1) // len(roots))
    buckets: List[List[Path]] = []
    for i, r in enumerate(roots):
        buckets.append(reservoir_sample_images(r, k=per_root_total, recurse=False, seed=seed + i + 1))

    combined: List[Path] = []
    for i in range(per_root_total):
        for b in buckets:
            if i < len(b):
                combined.append(b[i])

    combined = combined[:total]
    train = combined[:train_n]
    test = combined[train_n : train_n + test_n]
    return train, test


@dataclass
class Row:
    phase: str
    split: str
    sample_id: str
    image_path: str
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    retrieved_ids: List[str]
    vault_added_id: Optional[str]
    error: Optional[str] = None


def _avg(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def summarize(rows: List[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    add_flags = [r.teacher_should_add for r in rows if r.teacher_should_add is not None]
    return {
        "n": len(rows),
        "avg_teacher_score": _avg([float(x) for x in scores if x is not None]),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "add_rate_should_add": (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None,
    }


def run_split(
    *,
    pipeline: UrbanPipeline,
    phase: str,
    split: str,
    image_paths: List[Path],
    post_text: str,
) -> List[Row]:
    out: List[Row] = []
    for idx, image_path in enumerate(image_paths, start=1):
        sample_id = f"{split}_{idx}"
        sample = Sample(
            id=sample_id,
            image_path=str(image_path),
            post_text=post_text,
            ground_truth_sim=_dummy_gt(),
        )
        trace = pipeline.process_sample(sample)
        out.append(
            Row(
                phase=phase,
                split=split,
                sample_id=sample_id,
                image_path=str(image_path),
                teacher_score=float(trace["teacher_score"]) if isinstance(trace.get("teacher_score"), (int, float)) else None,
                teacher_should_add=bool(trace["teacher_should_add"]) if trace.get("teacher_should_add") is not None else None,
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                vault_added_id=trace.get("vault_added_id"),
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # Real backends
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")

    # Stable evaluation
    os.environ.setdefault("EVOSHOT_LLM_TEMPERATURE", "0")
    os.environ.setdefault("EVOSHOT_TEACHER_TEMPERATURE", "0")
    os.environ.setdefault("EVOSHOT_LLM_SEND_IMAGE", "1")
    os.environ.setdefault("EVOSHOT_LLM_USE_TOOLS", "auto")
    os.environ.setdefault("EVOSHOT_LLM_TOOL_CHOICE", "required")
    os.environ.setdefault("EVOSHOT_LLM_MAX_TOKENS", "512")

    # Defaults
    train_n = int(os.getenv("EVOSHOT_TRAIN_N", "10"))
    test_n = int(os.getenv("EVOSHOT_TEST_N", "10"))
    seed = int(os.getenv("EVOSHOT_SAMPLE_SEED", "42"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")

    roots = [
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\hanover"),
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\aachen"),
    ]
    for r in roots:
        if not r.exists():
            raise FileNotFoundError(f"Dataset root not found: {r}")

    train_imgs, test_imgs = sample_train_test_cityscapes(roots=roots, train_n=train_n, test_n=test_n, seed=seed)
    if not train_imgs or not test_imgs:
        raise RuntimeError("Failed to sample train/test images; check EVOSHOT_TRAIN_N/EVOSHOT_TEST_N.")

    # -------------------------
    # Phase A: baseline (no training, seeds only)
    # -------------------------
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = "caption+text"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = "2"
    os.environ["EVOSHOT_RETRIEVAL_NO_SELF_HIT"] = "0"

    baseline = UrbanPipeline()
    rows_baseline_test = run_split(
        pipeline=baseline,
        phase="baseline_seeds_only",
        split="test",
        image_paths=test_imgs,
        post_text=post_text,
    )

    # -------------------------
    # Phase B: train with E2 (caption+text + topk), then freeze
    # -------------------------
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = "caption+text"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = "2"
    os.environ["EVOSHOT_RETRIEVAL_NO_SELF_HIT"] = "0"

    trained = UrbanPipeline()
    rows_train_r1 = run_split(
        pipeline=trained,
        phase="train_r1_updates_on",
        split="train",
        image_paths=train_imgs,
        post_text=post_text,
    )

    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    # B1: re-eval train with self-hit allowed (measures caching upper bound)
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = "caption+text"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = "2"
    os.environ["EVOSHOT_RETRIEVAL_NO_SELF_HIT"] = "0"
    rows_train_r2_selfhit = run_split(
        pipeline=trained,
        phase="train_r2_selfhit_allowed",
        split="train",
        image_paths=train_imgs,
        post_text=post_text,
    )

    # B2: re-eval train with self-hit disabled (tests whether gains survive without exact answer recall)
    os.environ["EVOSHOT_RETRIEVAL_NO_SELF_HIT"] = "1"
    rows_train_r2_no_selfhit = run_split(
        pipeline=trained,
        phase="train_r2_no_selfhit",
        split="train",
        image_paths=train_imgs,
        post_text=post_text,
    )
    os.environ["EVOSHOT_RETRIEVAL_NO_SELF_HIT"] = "0"

    # B3: held-out test with similarity retrieval (what we actually want)
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = "caption+text"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = "2"
    rows_test_sim = run_split(
        pipeline=trained,
        phase="test_after_train_sim_topk",
        split="test",
        image_paths=test_imgs,
        post_text=post_text,
    )

    # B4: held-out test with RANDOM retrieval (controls for 'more text' vs 'right text')
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "random"
    os.environ["EVOSHOT_RANDOM_SEED"] = str(seed)
    rows_test_random = run_split(
        pipeline=trained,
        phase="test_after_train_random",
        split="test",
        image_paths=test_imgs,
        post_text=post_text,
    )

    # B5: held-out test with k=0 (no few-shot)
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = "0"
    rows_test_k0 = run_split(
        pipeline=trained,
        phase="test_after_train_k0",
        split="test",
        image_paths=test_imgs,
        post_text=post_text,
    )

    all_rows = (
        rows_baseline_test
        + rows_train_r1
        + rows_train_r2_selfhit
        + rows_train_r2_no_selfhit
        + rows_test_sim
        + rows_test_random
        + rows_test_k0
    )

    summaries = {
        "baseline_seeds_only_test": summarize(rows_baseline_test),
        "train_r1_updates_on": summarize(rows_train_r1),
        "train_r2_selfhit_allowed": summarize(rows_train_r2_selfhit),
        "train_r2_no_selfhit": summarize(rows_train_r2_no_selfhit),
        "test_after_train_sim_topk": summarize(rows_test_sim),
        "test_after_train_random": summarize(rows_test_random),
        "test_after_train_k0": summarize(rows_test_k0),
    }

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"learning_validation_{ts}.json"
    out_json.write_text(
        json.dumps(
            {
                "train_n": train_n,
                "test_n": test_n,
                "seed": seed,
                "train_images": [str(p) for p in train_imgs],
                "test_images": [str(p) for p in test_imgs],
                "summaries": summaries,
                "rows": [asdict(r) for r in all_rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"summaries": summaries, "results_path": str(out_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

