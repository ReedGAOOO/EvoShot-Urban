import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from evoshot_env import _load_dotenv
from models.filter import StudentUrbanFilter
from urban_experiment import Sample, UrbanPipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def reservoir_sample_images(root: Path, k: int, *, recurse: bool, seed: int, exclude: Optional[Set[Path]] = None) -> List[Path]:
    rng = random.Random(seed)
    reservoir: List[Path] = []
    seen = 0
    exclude = exclude or set()

    walker: Iterable[Path] = root.rglob("*") if recurse else root.glob("*")
    for p in walker:
        if not p.is_file():
            continue
        if not _is_image(p):
            continue
        try:
            resolved = p.resolve()
        except Exception:
            resolved = p
        if resolved in exclude:
            continue

        seen += 1
        if len(reservoir) < k:
            reservoir.append(resolved)
            continue
        j = rng.randrange(seen)
        if j < k:
            reservoir[j] = resolved

    return sorted(reservoir)


def _dummy_gt() -> dict:
    return {"safety": 3.0, "vibrancy": 3.0, "cleanliness": 3.0}


@dataclass
class Row:
    phase: str
    sample_id: str
    image_path: str
    retrieval_max_sim: Optional[float]
    evidence_facts: int
    rulebank_injected: Optional[bool]
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    seconds: float
    error: Optional[str] = None


def summarize(rows: Sequence[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    add_flags = [r.teacher_should_add for r in rows if r.teacher_should_add is not None]
    sims = [r.retrieval_max_sim for r in rows if isinstance(r.retrieval_max_sim, (int, float))]
    facts = [r.evidence_facts for r in rows if isinstance(r.evidence_facts, int)]
    evidence_used = [1 for r in rows if isinstance(r.evidence_facts, int) and r.evidence_facts > 0]
    secs = [r.seconds for r in rows if isinstance(r.seconds, (int, float))]
    rb_flags = [r.rulebank_injected for r in rows if isinstance(r.rulebank_injected, bool)]

    def avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": len(rows),
        "avg_teacher_score": avg([float(x) for x in scores if x is not None]),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "add_rate_should_add": (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None,
        "avg_retrieval_max_sim": avg([float(x) for x in sims if x is not None]),
        "avg_evidence_facts": avg([float(x) for x in facts]) if facts else None,
        "evidence_use_rate": (sum(evidence_used) / len(rows)) if rows else None,
        "rulebank_injected_rate": (sum(1 for x in rb_flags if x) / len(rb_flags)) if rb_flags else None,
        "avg_seconds": avg([float(x) for x in secs if x is not None]),
    }


def _set_env(env: Dict[str, str]) -> None:
    for k, v in env.items():
        os.environ[k] = v


def _select_train_images(
    *,
    root: Path,
    train_n: int,
    seed: int,
    recurse: bool,
    filter_urban: bool,
    min_conf: float,
    pool_factor: int,
    max_factor: int,
) -> tuple[List[Path], dict]:
    if train_n <= 0:
        return [], {"train_n": train_n, "selected": 0, "filter_urban": filter_urban}

    candidates_n = max(train_n * max(1, pool_factor), train_n)
    filterer = StudentUrbanFilter() if filter_urban else None

    chosen: List[Path] = []
    used_factor = max(1, pool_factor)
    attempts = 0
    while len(chosen) < train_n and used_factor <= max_factor:
        attempts += 1
        candidates_n = max(train_n * used_factor, train_n)
        candidates = reservoir_sample_images(root, k=candidates_n, recurse=recurse, seed=seed)
        chosen = []
        for p in candidates:
            if not filterer:
                chosen.append(p)
            else:
                try:
                    res = filterer.classify(str(p))
                except Exception:
                    continue
                if (not res.is_urban) or (res.confidence < min_conf):
                    continue
                chosen.append(p)
            if len(chosen) >= train_n:
                break
        if len(chosen) >= train_n:
            break
        used_factor *= 2

    stats = {
        "train_n": train_n,
        "selected": len(chosen[:train_n]),
        "filter_urban": filter_urban,
        "min_conf": min_conf,
        "pool_factor_start": pool_factor,
        "pool_factor_used": used_factor,
        "attempts": attempts,
        "root": str(root),
        "recurse": bool(recurse),
    }
    return chosen[:train_n], stats


def _select_test_images(
    *,
    root: Path,
    test_n: int,
    seed: int,
    recurse: bool,
    exclude: Optional[Set[Path]] = None,
) -> List[Path]:
    if test_n <= 0:
        return []
    return reservoir_sample_images(root, k=test_n, recurse=recurse, seed=seed, exclude=exclude)


def run_images(*, pipeline: UrbanPipeline, phase: str, prefix: str, image_paths: Sequence[Path], post_text: str) -> List[Row]:
    out: List[Row] = []
    for idx, image_path in enumerate(image_paths, start=1):
        sample_id = f"{prefix}_{idx}"
        sample = Sample(
            id=sample_id,
            image_path=str(image_path),
            post_text=post_text,
            ground_truth_sim=_dummy_gt(),
        )
        t0 = time.time()
        trace = pipeline.process_sample(sample)
        dt = max(0.0, time.time() - t0)

        meta = trace.get("retrieval_meta") or {}
        max_sim = meta.get("max_sim")
        try:
            max_sim_f = float(max_sim) if max_sim is not None else None
        except Exception:
            max_sim_f = None

        facts = trace.get("evidence_facts") or []
        n_facts = len(facts) if isinstance(facts, list) else 0

        injected = trace.get("rulebank_injected")
        injected_b = bool(injected) if injected is not None else None

        score = trace.get("teacher_score")
        score_f = float(score) if isinstance(score, (int, float)) else None
        error_text = str(trace.get("error")) if trace.get("error") else None
        # Treat runtime failures as zero-score outcomes (so averages aren't artificially inflated).
        if score_f is None and error_text:
            score_f = 0.0

        out.append(
            Row(
                phase=phase,
                sample_id=sample_id,
                image_path=str(image_path),
                retrieval_max_sim=max_sim_f,
                evidence_facts=n_facts,
                rulebank_injected=injected_b,
                teacher_score=score_f,
                teacher_should_add=bool(trace["teacher_should_add"]) if trace.get("teacher_should_add") is not None else None,
                seconds=dt,
                error=error_text,
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the current best-practice EvoShot-Urban pipeline on an image folder.")
    ap.add_argument("--image_dir", required=True, help="Folder containing images")
    ap.add_argument("--train_dir", default=None, help="Optional separate train folder (defaults to --image_dir)")
    ap.add_argument("--test_dir", default=None, help="Optional separate test folder (defaults to --image_dir)")
    ap.add_argument("--train_n", type=int, default=8, help="Number of training images (updates ON)")
    ap.add_argument("--test_n", type=int, default=16, help="Number of test images (updates OFF)")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed")
    ap.add_argument("--recurse", action="store_true", help="Recursively scan folders for images")
    ap.add_argument("--no_filter_train", action="store_true", help="Disable StudentUrbanFilter on training selection")
    ap.add_argument("--filter_min_conf", type=float, default=0.55, help="Urban filter min confidence")
    ap.add_argument("--filter_pool_factor", type=int, default=8, help="Candidate pool multiplier for filtering")
    ap.add_argument("--filter_max_factor", type=int, default=40, help="Max pool multiplier for filtering retries")
    ap.add_argument("--allow_overlap", action="store_true", help="Allow train/test overlap when using one folder")
    ap.add_argument("--post_text", default="Please score this image.", help="Post text for all samples")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    train_dir = Path(args.train_dir).expanduser().resolve() if args.train_dir else image_dir
    test_dir = Path(args.test_dir).expanduser().resolve() if args.test_dir else image_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")

    # -------------------------
    # Best-practice defaults (override behavior knobs, but do NOT override endpoint/model secrets)
    # -------------------------
    _set_env(
        {
            # Retrieval (stable trunk)
            "EVOSHOT_RETRIEVAL_STRATEGY": "topk",
            "EVOSHOT_RETRIEVAL_K": "2",
            "EVOSHOT_RETRIEVAL_NO_SELF_HIT": "1",
            "EVOSHOT_STUDENT_FEWSHOT_MODE": "text",
            "EVOSHOT_VAULT_EMBED_MODE": "text",
            "EVOSHOT_QUERY_EMBED_MODE": "caption+text",
            "EVOSHOT_VAULT_CAPTION_BACKEND": "student",
            # RuleBank
            "EVOSHOT_RULEBANK_ENABLED": "1",
            "EVOSHOT_RULES_K": "4",
            "EVOSHOT_RULEBANK_GATE": "always",
            # Evidence (gated by retrieval_fail_or_low_sim, tuned threshold)
            "EVOSHOT_EVIDENCE_ENABLED": "1",
            "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
            "EVOSHOT_EVIDENCE_MIN_SIM": "0.68",
            # Keep advanced knobs off by default (as per ablations)
            "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
            "EVOSHOT_RETRIEVAL_QUERY_FUSION": "0",
            "EVOSHOT_SHOTS_FORMAT": "full",
            "EVOSHOT_SHOTS_MIN_SIM": "",
            "EVOSHOT_STUDENT_VISUAL_EVIDENCE": "0",
            "EVOSHOT_SELF_REVISION_ENABLED": "0",
            # Determinism (as much as possible)
            "EVOSHOT_LLM_TEMPERATURE": "0",
            "EVOSHOT_TEACHER_TEMPERATURE": "0",
            "EVOSHOT_LLM_SEND_IMAGE": "1",
            "EVOSHOT_LLM_USE_TOOLS": "auto",
            "EVOSHOT_LLM_TOOL_CHOICE": "required",
            "EVOSHOT_LLM_MAX_TOKENS": "512",
            "EVOSHOT_RANDOM_SEED": str(args.seed),
        }
    )

    # -------------------------
    # Select train/test sets
    # -------------------------
    train_imgs, filter_stats = _select_train_images(
        root=train_dir,
        train_n=max(0, int(args.train_n)),
        seed=int(args.seed),
        recurse=bool(args.recurse),
        filter_urban=not bool(args.no_filter_train),
        min_conf=float(args.filter_min_conf),
        pool_factor=int(args.filter_pool_factor),
        max_factor=int(args.filter_max_factor),
    )
    exclude: Set[Path] = set(p.resolve() for p in train_imgs) if (train_dir == test_dir and not args.allow_overlap) else set()
    test_imgs = _select_test_images(
        root=test_dir,
        test_n=max(0, int(args.test_n)),
        seed=int(args.seed) + 100,
        recurse=bool(args.recurse),
        exclude=exclude,
    )

    # -------------------------
    # Train once (updates ON) but keep prompts clean
    # -------------------------
    _set_env(
        {
            "EVOSHOT_DISABLE_VAULT_UPDATE": "0",
            "EVOSHOT_RULEBANK_ENABLED": "0",
            "EVOSHOT_EVIDENCE_ENABLED": "0",
        }
    )
    pipeline = UrbanPipeline()
    rows_train = run_images(pipeline=pipeline, phase="train_updates_on", prefix="train", image_paths=train_imgs, post_text=args.post_text)

    # -------------------------
    # Evaluate (updates OFF) with best-practice trunk config
    # -------------------------
    _set_env(
        {
            "EVOSHOT_DISABLE_VAULT_UPDATE": "1",
            "EVOSHOT_RULEBANK_ENABLED": "1",
            "EVOSHOT_EVIDENCE_ENABLED": "1",
        }
    )
    rows_test = run_images(pipeline=pipeline, phase="test_best_practice", prefix="test", image_paths=test_imgs, post_text=args.post_text)

    out = {
        "script": Path(__file__).name,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": vars(args),
        "env_best_practice": {
            "EVOSHOT_RETRIEVAL_STRATEGY": os.getenv("EVOSHOT_RETRIEVAL_STRATEGY"),
            "EVOSHOT_RETRIEVAL_K": os.getenv("EVOSHOT_RETRIEVAL_K"),
            "EVOSHOT_RULES_K": os.getenv("EVOSHOT_RULES_K"),
            "EVOSHOT_EVIDENCE_GATE": os.getenv("EVOSHOT_EVIDENCE_GATE"),
            "EVOSHOT_EVIDENCE_MIN_SIM": os.getenv("EVOSHOT_EVIDENCE_MIN_SIM"),
            "EVOSHOT_QUERY_EMBED_MODE": os.getenv("EVOSHOT_QUERY_EMBED_MODE"),
            "EVOSHOT_VAULT_EMBED_MODE": os.getenv("EVOSHOT_VAULT_EMBED_MODE"),
            "EVOSHOT_STUDENT_FEWSHOT_MODE": os.getenv("EVOSHOT_STUDENT_FEWSHOT_MODE"),
        },
        "filter_stats": filter_stats,
        "train_images": [str(p) for p in train_imgs],
        "test_images": [str(p) for p in test_imgs],
        "summaries": {
            "train_updates_on": summarize(rows_train),
            "test_best_practice": summarize(rows_test),
        },
        "rows": [asdict(r) for r in (list(rows_train) + list(rows_test))],
    }

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out"))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"best_practice_folder_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"results_path": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

