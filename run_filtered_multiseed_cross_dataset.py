import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, List, Optional

from evoshot_env import _load_dotenv
from models.filter import StudentUrbanFilter
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


@dataclass
class Row:
    seed: int
    phase: str
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


def run_images(
    *,
    seed: int,
    pipeline: UrbanPipeline,
    phase: str,
    prefix: str,
    image_paths: List[Path],
    post_text: str,
) -> List[Row]:
    out: List[Row] = []
    for idx, image_path in enumerate(image_paths, start=1):
        sample_id = f"{prefix}_{idx}"
        sample = Sample(
            id=sample_id,
            image_path=str(image_path),
            post_text=post_text,
            ground_truth_sim=_dummy_gt(),
        )
        trace = pipeline.process_sample(sample)
        out.append(
            Row(
                seed=seed,
                phase=phase,
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


def _parse_seeds(text: str) -> List[int]:
    parts: List[int] = []
    for chunk in (text or "").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            parts.append(int(chunk))
        except Exception:
            continue
    return parts


def _aggregate(seed_summaries: List[dict], key: str) -> dict:
    vals = []
    for s in seed_summaries:
        v = (s.get(key) or {}).get("avg_teacher_score")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return {
        "n_seeds": len(seed_summaries),
        "avg_teacher_score_mean": mean(vals) if vals else None,
        "avg_teacher_score_stdev": pstdev(vals) if len(vals) > 1 else None,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # --- Use original text-only context form (no image embedding, no multimodal few-shot) ---
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_STUDENT_FEWSHOT_MODE", "text")
    os.environ.setdefault("EVOSHOT_VAULT_EMBED_MODE", "text")
    os.environ.setdefault("EVOSHOT_QUERY_EMBED_MODE", "caption+text")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_K", "2")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_NO_SELF_HIT", "1")
    os.environ.setdefault("EVOSHOT_VAULT_CAPTION_BACKEND", "student")  # avoid extra/unstable teacher caption calls

    # Stability
    os.environ.setdefault("EVOSHOT_LLM_TEMPERATURE", "0")
    os.environ.setdefault("EVOSHOT_TEACHER_TEMPERATURE", "0")
    os.environ.setdefault("EVOSHOT_LLM_SEND_IMAGE", "1")
    os.environ.setdefault("EVOSHOT_LLM_USE_TOOLS", "auto")
    os.environ.setdefault("EVOSHOT_LLM_TOOL_CHOICE", "required")
    os.environ.setdefault("EVOSHOT_LLM_MAX_TOKENS", "512")

    train_n = int(os.getenv("EVOSHOT_TRAIN_N", "10"))
    test_n = int(os.getenv("EVOSHOT_TEST_N", "10"))
    batch_seed = int(os.getenv("EVOSHOT_BATCH_SEED", "42"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")
    retrieval_k = int(os.getenv("EVOSHOT_RETRIEVAL_K", "2"))
    retrieval_k = max(0, retrieval_k)

    seeds = _parse_seeds(os.getenv("EVOSHOT_EVAL_SEEDS", "0,1,2"))

    pool_factor = int(os.getenv("EVOSHOT_FILTER_POOL_FACTOR", "8"))
    min_conf = float(os.getenv("EVOSHOT_FILTER_MIN_CONF", "0.55"))
    max_factor = int(os.getenv("EVOSHOT_FILTER_MAX_FACTOR", "40"))

    train_root = Path(r"H:\微博数据 - 副本\images")
    test_roots = [
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\hanover"),
        Path(r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\aachen"),
    ]
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    for r in test_roots:
        if not r.exists():
            raise FileNotFoundError(f"Test root not found: {r}")

    # -------------------------
    # Fixed train/test batch selection
    # -------------------------
    if train_n <= 0:
        train_imgs: List[Path] = []
        filter_stats = {"train_n": train_n, "selected": 0, "pool_factor": 0, "min_conf": min_conf}
    else:
        print(f"[FILTER] Selecting {train_n} urban images from dataset1 (batch_seed={batch_seed})...")
        filterer = StudentUrbanFilter()
        chosen: List[Path] = []
        used_factor = pool_factor
        attempts = 0
        while len(chosen) < train_n and used_factor <= max_factor:
            attempts += 1
            cand_n = max(train_n * used_factor, train_n)
            print(f"[FILTER] Attempt {attempts}: sampling {cand_n} candidates (factor={used_factor})")
            candidates = reservoir_sample_images(train_root, k=cand_n, recurse=True, seed=batch_seed)
            chosen = []
            for p in candidates:
                try:
                    res = filterer.classify(str(p))
                except Exception:
                    continue
                if (not res.is_urban) or (res.confidence < min_conf):
                    continue
                chosen.append(p)
                if len(chosen) % 5 == 0:
                    print(f"[FILTER] accepted={len(chosen)}/{train_n}")
                if len(chosen) >= train_n:
                    break
            if len(chosen) >= train_n:
                break
            used_factor *= 2
        train_imgs = chosen[:train_n]
        filter_stats = {
            "train_n": train_n,
            "selected": len(train_imgs),
            "pool_factor_start": pool_factor,
            "pool_factor_used": used_factor,
            "attempts": attempts,
            "min_conf": min_conf,
        }
        print(f"[FILTER] Done: selected {len(train_imgs)}/{train_n} | stats={filter_stats}")

    per_root = max(1, (test_n + len(test_roots) - 1) // len(test_roots)) if test_n > 0 else 0
    test_imgs: List[Path] = []
    for i, r in enumerate(test_roots):
        if per_root <= 0:
            continue
        test_imgs.extend(reservoir_sample_images(r, k=per_root, recurse=False, seed=batch_seed + 100 + i))
    test_imgs = test_imgs[:test_n]
    print(f"[DATA] Fixed test set: {len(test_imgs)}/{test_n} images")

    # -------------------------
    # Multi-seed runs on the fixed batch
    # -------------------------
    seed_payloads: List[dict] = []
    all_rows: List[Row] = []

    # Baseline once (seeds only)
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = "caption+text"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
    baseline = UrbanPipeline()
    rows_baseline_test = run_images(
        seed=-1,
        pipeline=baseline,
        phase="baseline_seeds_only_test",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    baseline_summary = summarize(rows_baseline_test)
    print(f"[BASELINE] seeds-only test summary: {baseline_summary}")
    all_rows.extend(rows_baseline_test)

    # Train once
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
    trained = UrbanPipeline()
    rows_train = run_images(
        seed=-1,
        pipeline=trained,
        phase="train_updates_on",
        prefix="train",
        image_paths=train_imgs,
        post_text=post_text,
    )
    train_summary = summarize(rows_train)
    print(f"[TRAIN] updates-on summary: {train_summary}")
    all_rows.extend(rows_train)

    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    # For each eval seed: compare sim-topk vs random vs k=0 on the same trained vault
    for s in seeds:
        print(f"\n[SEED {s}] Evaluating on fixed test set...")

        os.environ["EVOSHOT_RANDOM_SEED"] = str(s)

        os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
        os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
        rows_test_sim = run_images(seed=s, pipeline=trained, phase="test_after_train_sim_topk", prefix="test", image_paths=test_imgs, post_text=post_text)

        os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "random"
        os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
        rows_test_random = run_images(seed=s, pipeline=trained, phase="test_after_train_random", prefix="test", image_paths=test_imgs, post_text=post_text)

        os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
        os.environ["EVOSHOT_RETRIEVAL_K"] = "0"
        rows_test_k0 = run_images(seed=s, pipeline=trained, phase="test_after_train_k0", prefix="test", image_paths=test_imgs, post_text=post_text)

        seed_summary = {
            "seed": s,
            "baseline_seeds_only_test": baseline_summary,
            "train_updates_on": train_summary,
            "test_after_train_sim_topk": summarize(rows_test_sim),
            "test_after_train_random": summarize(rows_test_random),
            "test_after_train_k0": summarize(rows_test_k0),
        }
        print(f"[SEED {s}] sim_topk={seed_summary['test_after_train_sim_topk']} | random={seed_summary['test_after_train_random']} | k0={seed_summary['test_after_train_k0']}")

        seed_payloads.append(seed_summary)
        all_rows.extend(rows_test_sim)
        all_rows.extend(rows_test_random)
        all_rows.extend(rows_test_k0)

    aggregated = {
        "baseline_seeds_only_test": _aggregate(seed_payloads, "baseline_seeds_only_test"),
        "train_updates_on": _aggregate(seed_payloads, "train_updates_on"),
        "test_after_train_sim_topk": _aggregate(seed_payloads, "test_after_train_sim_topk"),
        "test_after_train_random": _aggregate(seed_payloads, "test_after_train_random"),
        "test_after_train_k0": _aggregate(seed_payloads, "test_after_train_k0"),
    }

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"filtered_multiseed_cross_{ts}.json"
    out_json.write_text(
        json.dumps(
            {
                "train_n": train_n,
                "test_n": test_n,
                "batch_seed": batch_seed,
                "eval_seeds": seeds,
                "filter_stats": filter_stats,
                "train_images": [str(p) for p in train_imgs],
                "test_images": [str(p) for p in test_imgs],
                "seed_summaries": seed_payloads,
                "aggregated": aggregated,
                "rows": [asdict(r) for r in all_rows],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"aggregated": aggregated, "results_path": str(out_json)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
