import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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
    phase: str
    sample_id: str
    image_path: str
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    retrieved_ids: List[str]
    rule_ids: List[str]
    vault_added_id: Optional[str]
    rule_added_id: Optional[str]
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
                phase=phase,
                sample_id=sample_id,
                image_path=str(image_path),
                teacher_score=float(trace["teacher_score"]) if isinstance(trace.get("teacher_score"), (int, float)) else None,
                teacher_should_add=bool(trace["teacher_should_add"]) if trace.get("teacher_should_add") is not None else None,
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                rule_ids=list(trace.get("rule_ids") or []),
                vault_added_id=trace.get("vault_added_id"),
                rule_added_id=trace.get("rule_added_id"),
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # --- Original text-only context form (no image embedding, no multimodal few-shot) ---
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_STUDENT_FEWSHOT_MODE", "text")
    os.environ.setdefault("EVOSHOT_VAULT_EMBED_MODE", "text")
    os.environ.setdefault("EVOSHOT_QUERY_EMBED_MODE", "caption+text")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_STRATEGY", "topk")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_K", "2")
    os.environ.setdefault("EVOSHOT_RETRIEVAL_NO_SELF_HIT", "1")
    os.environ.setdefault("EVOSHOT_VAULT_CAPTION_BACKEND", "student")

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
    rules_k = int(os.getenv("EVOSHOT_RULES_K", "4"))
    rules_k = max(0, rules_k)

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
    # Train once to build Vault + RuleBank
    # -------------------------
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    os.environ["EVOSHOT_RULEBANK_ENABLED"] = "0"  # rulebank initially empty; keep prompt unchanged during training
    os.environ["EVOSHOT_RULES_K"] = str(rules_k)
    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
    pipeline = UrbanPipeline()

    rows_train = run_images(
        pipeline=pipeline,
        phase="train_updates_on",
        prefix="train",
        image_paths=train_imgs,
        post_text=post_text,
    )
    train_summary = summarize(rows_train)
    rulebank_size = len(getattr(pipeline, "rulebank", None).rules) if getattr(pipeline, "rulebank", None) is not None else None
    print(f"[TRAIN] summary: {train_summary} | rulebank_size={rulebank_size}")

    # Freeze learning (both vault and rulebank updates)
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    # -------------------------
    # Evaluate: k=0 vs sim-topk, with/without RuleBank
    # -------------------------
    rows_all: List[Row] = []
    rows_all.extend(rows_train)

    phases = [
        ("k0_no_rules", {"EVOSHOT_RETRIEVAL_K": "0", "EVOSHOT_RULEBANK_ENABLED": "0"}),
        ("k0_with_rules", {"EVOSHOT_RETRIEVAL_K": "0", "EVOSHOT_RULEBANK_ENABLED": "1"}),
        ("sim_topk_no_rules", {"EVOSHOT_RETRIEVAL_K": str(retrieval_k), "EVOSHOT_RULEBANK_ENABLED": "0"}),
        ("sim_topk_with_rules", {"EVOSHOT_RETRIEVAL_K": str(retrieval_k), "EVOSHOT_RULEBANK_ENABLED": "1"}),
    ]

    phase_summaries = {"train_updates_on": train_summary}
    for phase, env in phases:
        for k, v in env.items():
            os.environ[k] = v
        rows = run_images(
            pipeline=pipeline,
            phase=phase,
            prefix=phase,
            image_paths=test_imgs,
            post_text=post_text,
        )
        phase_summaries[phase] = summarize(rows)
        rows_all.extend(rows)
        print(f"[EVAL] {phase}: {phase_summaries[phase]}")

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"rulebank_ablation_cross_{ts}.json"
    out_json.write_text(
        json.dumps(
            {
                "train_n": train_n,
                "test_n": test_n,
                "batch_seed": batch_seed,
                "retrieval_k": retrieval_k,
                "rules_k": rules_k,
                "filter_stats": filter_stats,
                "train_images": [str(p) for p in train_imgs],
                "test_images": [str(p) for p in test_imgs],
                "rulebank_size_after_train": rulebank_size,
                "summaries": phase_summaries,
                "rows": [asdict(r) for r in rows_all],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"results_path": str(out_json), "summaries": phase_summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

