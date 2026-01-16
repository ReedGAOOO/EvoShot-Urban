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
    retrieval_max_sim: Optional[float]
    retrieved_ids: List[str]
    teacher_score: Optional[float]
    pass_ge_0_8: Optional[bool]
    evidence_facts: int
    seconds: float
    error: Optional[str] = None


def summarize(rows: List[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    sims = [r.retrieval_max_sim for r in rows if isinstance(r.retrieval_max_sim, (int, float))]
    facts = [r.evidence_facts for r in rows if isinstance(r.evidence_facts, int)]
    used = [1 for r in rows if isinstance(r.evidence_facts, int) and r.evidence_facts > 0]
    secs = [r.seconds for r in rows if isinstance(r.seconds, (int, float))]

    def avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": len(rows),
        "avg_teacher_score": avg([float(x) for x in scores if x is not None]),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "avg_retrieval_max_sim": avg([float(x) for x in sims if x is not None]),
        "avg_evidence_facts": avg([float(x) for x in facts]) if facts else None,
        "evidence_use_rate": (sum(used) / len(rows)) if rows else None,
        "avg_seconds": avg([float(x) for x in secs]) if secs else None,
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

        score = trace.get("teacher_score")
        score_f = float(score) if isinstance(score, (int, float)) else None

        out.append(
            Row(
                phase=phase,
                sample_id=sample_id,
                image_path=str(image_path),
                retrieval_max_sim=max_sim_f,
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                teacher_score=score_f,
                pass_ge_0_8=(score_f is not None and score_f >= 0.8),
                evidence_facts=n_facts,
                seconds=dt,
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def _grid_from_max_sims(max_sims: List[float]) -> List[float]:
    if not max_sims:
        return [0.2, 0.35, 0.5]
    eps = 1e-6
    uniq = sorted(set(float(x) for x in max_sims))
    grid = {0.0, 1.0}
    for v in uniq:
        grid.add(v)
        grid.add(min(1.0, v + eps))
    return sorted(grid)


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # --- Pipeline defaults (best-known framework) ---
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

    train_n = int(os.getenv("EVOSHOT_TRAIN_N", "8"))
    test_n = int(os.getenv("EVOSHOT_TEST_N", "16"))
    batch_seed = int(os.getenv("EVOSHOT_BATCH_SEED", "43"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")
    retrieval_k = max(0, int(os.getenv("EVOSHOT_RETRIEVAL_K", "2")))
    rules_k = max(0, int(os.getenv("EVOSHOT_RULES_K", "4")))

    baseline_min_sim = float(os.getenv("EVOSHOT_EVIDENCE_MIN_SIM", "0.68"))

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
    # Fixed train/test selection (dataset1 is noisy -> filter by Student)
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

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"structured_caption_min_sim_calibration_cross_{ts}.json"

    # ------------------------------------------------------------
    # Train once (learning on): build Vault + RuleBank, but keep prompts "clean"
    # ------------------------------------------------------------
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    os.environ["EVOSHOT_RULEBANK_ENABLED"] = "0"
    os.environ["EVOSHOT_RULES_K"] = str(rules_k)
    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "0"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "always"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_FUSION"] = "0"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_EXPANSION"] = "0"
    os.environ["EVOSHOT_RETRIEVAL_CAPTION_STYLE"] = "sentence"
    os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)

    pipeline = UrbanPipeline()
    rows_train = run_images(pipeline=pipeline, phase="train_updates_on", prefix="train", image_paths=train_imgs, post_text=post_text)
    train_summary = summarize(rows_train)
    print(f"[TRAIN] summary: {train_summary}")

    # Freeze learning
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    # ------------------------------------------------------------
    # Baseline(best): sentence caption + gate(min_sim=0.68)
    # ------------------------------------------------------------
    os.environ["EVOSHOT_RULEBANK_ENABLED"] = "1"
    os.environ["EVOSHOT_RULES_K"] = str(rules_k)
    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "1"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "fail_or_low_sim"
    os.environ["EVOSHOT_EVIDENCE_MIN_SIM"] = str(baseline_min_sim)
    os.environ["EVOSHOT_RETRIEVAL_CAPTION_STYLE"] = "sentence"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_FUSION"] = "0"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_EXPANSION"] = "0"
    rows_baseline = run_images(
        pipeline=pipeline,
        phase="baseline_sentence_gate",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    baseline_summary = summarize(rows_baseline)
    target_ev_rate = float(baseline_summary.get("evidence_use_rate") or 0.0)
    print(f"[BASELINE] {baseline_summary}")

    # ------------------------------------------------------------
    # Structured caption: collect base(no evidence) + evidence(always)
    # ------------------------------------------------------------
    os.environ["EVOSHOT_RETRIEVAL_CAPTION_STYLE"] = "structured"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_FUSION"] = "0"
    os.environ["EVOSHOT_RETRIEVAL_QUERY_EXPANSION"] = "0"

    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "0"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "always"
    rows_struct_base = run_images(
        pipeline=pipeline,
        phase="structured_no_evidence",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    struct_base_summary = summarize(rows_struct_base)
    print(f"[STRUCT base] {struct_base_summary}")

    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "1"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "always"
    rows_struct_ev = run_images(
        pipeline=pipeline,
        phase="structured_evidence_always",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    struct_ev_summary = summarize(rows_struct_ev)
    print(f"[STRUCT evidence(always)] {struct_ev_summary}")

    # -------------------------
    # Offline sweep: pick threshold that matches baseline evidence_use_rate
    # -------------------------
    base_by_id = {r.sample_id: r for r in rows_struct_base}
    ev_by_id = {r.sample_id: r for r in rows_struct_ev}

    max_sims = [r.retrieval_max_sim for r in rows_struct_base if isinstance(r.retrieval_max_sim, (int, float))]
    grid = _grid_from_max_sims([float(x) for x in max_sims])

    sweep: List[dict] = []
    for t in grid:
        used = 0
        used_n = 0
        scores: List[float] = []
        secs: List[float] = []
        for sid, base_row in base_by_id.items():
            ev_row = ev_by_id.get(sid)
            if ev_row is None:
                continue

            retrieval_failed = (retrieval_k <= 0) or (len(base_row.retrieved_ids) == 0)
            ms = base_row.retrieval_max_sim
            low_sim = (ms is None) or (float(ms) < float(t))
            use_ev = retrieval_failed or low_sim

            score = ev_row.teacher_score if use_ev else base_row.teacher_score
            sec = ev_row.seconds if use_ev else base_row.seconds
            if score is None:
                continue
            used_n += 1
            scores.append(float(score))
            secs.append(float(sec))
            if use_ev:
                used += 1

        avg_score = (sum(scores) / len(scores)) if scores else None
        pass_rate = (sum(1 for s in scores if s >= 0.8) / len(scores)) if scores else None
        ev_rate = (used / used_n) if used_n else None
        avg_sec = (sum(secs) / len(secs)) if secs else None
        sweep.append(
            {
                "min_sim": float(t),
                "n_used": int(used_n),
                "avg_teacher_score": avg_score,
                "pass_rate_ge_0_8": pass_rate,
                "evidence_use_rate": ev_rate,
                "avg_seconds": avg_sec,
                "abs_diff_evidence_use_rate_vs_baseline": abs(float(ev_rate) - target_ev_rate) if ev_rate is not None else None,
            }
        )

    def _key(x: dict) -> tuple:
        diff = x.get("abs_diff_evidence_use_rate_vs_baseline")
        score = x.get("avg_teacher_score")
        return (
            float(diff) if diff is not None else 1e9,
            -(float(score) if score is not None else -1e9),
            float(x.get("avg_seconds") or 1e9),
        )

    sweep_sorted = sorted(sweep, key=_key)
    best = sweep_sorted[0] if sweep_sorted else None
    print(f"[CALIB] target_evidence_use_rate={target_ev_rate} best={best}")

    # ------------------------------------------------------------
    # Validate: run actual structured caption gating with tuned threshold
    # ------------------------------------------------------------
    rows_struct_tuned: List[Row] = []
    tuned_summary = None
    if best is not None and best.get("min_sim") is not None:
        os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "1"
        os.environ["EVOSHOT_EVIDENCE_GATE"] = "fail_or_low_sim"
        os.environ["EVOSHOT_EVIDENCE_MIN_SIM"] = str(best["min_sim"])
        rows_struct_tuned = run_images(
            pipeline=pipeline,
            phase=f"structured_gate_tuned_{best['min_sim']}",
            prefix="test",
            image_paths=test_imgs,
            post_text=post_text,
        )
        tuned_summary = summarize(rows_struct_tuned)
        print(f"[VALIDATE tuned] {tuned_summary}")

    out_json.write_text(
        json.dumps(
            {
                "train_n": train_n,
                "test_n": test_n,
                "batch_seed": batch_seed,
                "retrieval_k": retrieval_k,
                "rules_k": rules_k,
                "baseline_min_sim": baseline_min_sim,
                "baseline_summary": baseline_summary,
                "structured_base_summary": struct_base_summary,
                "structured_evidence_always_summary": struct_ev_summary,
                "target_evidence_use_rate": target_ev_rate,
                "calibration_grid_n": len(grid),
                "sweep_sorted": sweep_sorted,
                "best": best,
                "validate_tuned_summary": tuned_summary,
                "filter_stats": filter_stats,
                "train_images": [str(p) for p in train_imgs],
                "test_images": [str(p) for p in test_imgs],
                "rows_train": [asdict(r) for r in rows_train],
                "rows_baseline": [asdict(r) for r in rows_baseline],
                "rows_struct_base": [asdict(r) for r in rows_struct_base],
                "rows_struct_evidence_always": [asdict(r) for r in rows_struct_ev],
                "rows_struct_tuned": [asdict(r) for r in rows_struct_tuned],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"results_path": str(out_json), "best": best, "validate_tuned_summary": tuned_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

