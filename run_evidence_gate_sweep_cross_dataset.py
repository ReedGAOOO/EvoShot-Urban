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


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def _parse_float_list(v: str) -> List[float]:
    out: List[float] = []
    for part in (v or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return out


def _linspace(start: float, end: float, step: float) -> List[float]:
    start_f = float(start)
    end_f = float(end)
    step_f = float(step)
    if step_f <= 0:
        raise ValueError("step must be > 0")
    if end_f < start_f:
        start_f, end_f = end_f, start_f
    out: List[float] = []
    x = start_f
    # Include end (with tolerance).
    while x <= end_f + 1e-12:
        out.append(float(x))
        x += step_f
    return out


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        raise ValueError("quantile on empty list")
    q = max(0.0, min(1.0, float(q)))
    xs_sorted = sorted(xs)
    if len(xs_sorted) == 1:
        return float(xs_sorted[0])
    pos = q * (len(xs_sorted) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs_sorted) - 1)
    frac = pos - lo
    return float(xs_sorted[lo] * (1.0 - frac) + xs_sorted[hi] * frac)


@dataclass
class Row:
    phase: str
    sample_id: str
    image_path: str
    retrieval_max_sim: Optional[float]
    retrieved_ids: List[str]
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    evidence_facts: int
    seconds: float
    error: Optional[str] = None


def summarize(rows: List[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    add_flags = [r.teacher_should_add for r in rows if r.teacher_should_add is not None]
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
        "add_rate_should_add": (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None,
        "avg_retrieval_max_sim": avg([float(x) for x in sims if x is not None]),
        "avg_evidence_facts": avg([float(x) for x in facts]) if facts else None,
        "evidence_use_rate": (sum(used) / len(rows)) if rows else None,
        "avg_seconds": avg([float(x) for x in secs if x is not None]),
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

        out.append(
            Row(
                phase=phase,
                sample_id=sample_id,
                image_path=str(image_path),
                retrieval_max_sim=max_sim_f,
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                teacher_score=float(trace["teacher_score"]) if isinstance(trace.get("teacher_score"), (int, float)) else None,
                teacher_should_add=bool(trace["teacher_should_add"]) if trace.get("teacher_should_add") is not None else None,
                evidence_facts=n_facts,
                seconds=dt,
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # --- Pipeline defaults (can be overridden via env/.env) ---
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

    # --- Fixed train/test batch selection ---
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

    # --- Train once to build Vault + RuleBank ---
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "0"
    os.environ["EVOSHOT_RULEBANK_ENABLED"] = "0"
    os.environ["EVOSHOT_RULES_K"] = str(rules_k)
    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "0"
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

    # Freeze learning
    os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

    # --- Collect two reference runs on the SAME test set ---
    # Base: no evidence
    os.environ["EVOSHOT_RULEBANK_ENABLED"] = "1"
    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "0"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "always"
    os.environ["EVOSHOT_RETRIEVAL_K"] = str(retrieval_k)
    rows_base = run_images(
        pipeline=pipeline,
        phase="sim_topk_with_rules_no_evidence",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    base_summary = summarize(rows_base)
    print(f"[BASE] {base_summary}")

    # Ref: always evidence
    os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "1"
    os.environ["EVOSHOT_EVIDENCE_GATE"] = "always"
    rows_ev = run_images(
        pipeline=pipeline,
        phase="sim_topk_with_rules_with_evidence",
        prefix="test",
        image_paths=test_imgs,
        post_text=post_text,
    )
    ev_summary = summarize(rows_ev)
    print(f"[EVIDENCE(always)] {ev_summary}")

    # --- Sweep thresholds offline (mixture of base/evidence by max_sim) ---
    base_by_id = {r.sample_id: r for r in rows_base}
    ev_by_id = {r.sample_id: r for r in rows_ev}

    max_sims = [r.retrieval_max_sim for r in rows_base if isinstance(r.retrieval_max_sim, (int, float))]
    grid_env = os.getenv("EVOSHOT_EVIDENCE_MIN_SIM_GRID", "").strip()
    start_env = os.getenv("EVOSHOT_EVIDENCE_MIN_SIM_START", "").strip()
    end_env = os.getenv("EVOSHOT_EVIDENCE_MIN_SIM_END", "").strip()
    step_env = os.getenv("EVOSHOT_EVIDENCE_MIN_SIM_STEP", "").strip()
    if step_env:
        start = float(start_env) if start_env else 0.0
        end = float(end_env) if end_env else 1.0
        step = float(step_env)
        grid = [round(x, 4) for x in _linspace(start, end, step)]
    elif grid_env:
        grid = sorted(set(_parse_float_list(grid_env)))
    elif max_sims:
        qs = [0.2, 0.4, 0.6, 0.8]
        grid = sorted({round(_quantile(max_sims, q), 4) for q in qs})
    else:
        grid = [0.25, 0.35, 0.45]

    sweep_results: List[dict] = []
    for t in grid:
        chosen_scores: List[float] = []
        chosen_secs: List[float] = []
        used_evidence = 0
        used_n = 0
        for sid, base_row in base_by_id.items():
            ev_row = ev_by_id.get(sid)
            if ev_row is None:
                continue

            retrieval_failed = (retrieval_k <= 0) or (len(base_row.retrieved_ids) == 0)
            ms = base_row.retrieval_max_sim
            low_sim = (ms is None) or (float(ms) < float(t))
            use_ev = retrieval_failed or low_sim

            score = ev_row.teacher_score if use_ev else base_row.teacher_score
            secs = ev_row.seconds if use_ev else base_row.seconds
            if score is None:
                score = base_row.teacher_score if use_ev else ev_row.teacher_score
            if score is None:
                continue

            used_n += 1
            chosen_scores.append(float(score))
            chosen_secs.append(float(secs))
            if use_ev:
                used_evidence += 1

        avg_score = (sum(chosen_scores) / len(chosen_scores)) if chosen_scores else None
        pass_rate = (sum(1 for s in chosen_scores if s >= 0.8) / len(chosen_scores)) if chosen_scores else None
        avg_secs = (sum(chosen_secs) / len(chosen_secs)) if chosen_secs else None
        evidence_rate = (used_evidence / used_n) if used_n else None
        sweep_results.append(
            {
                "min_sim": float(t),
                "n_used": int(used_n),
                "avg_teacher_score": avg_score,
                "pass_rate_ge_0_8": pass_rate,
                "evidence_use_rate": evidence_rate,
                "avg_seconds": avg_secs,
                "delta_vs_base": (avg_score - base_summary["avg_teacher_score"])
                if (avg_score is not None and base_summary.get("avg_teacher_score") is not None)
                else None,
            }
        )

    # Sort: higher avg_teacher_score first; if tie, lower evidence_use_rate first.
    sweep_results_sorted = sorted(
        sweep_results,
        key=lambda x: (
            -(float(x["avg_teacher_score"]) if x.get("avg_teacher_score") is not None else -1e9),
            float(x["evidence_use_rate"]) if x.get("evidence_use_rate") is not None else 1e9,
        ),
    )

    # Pick best: max avg_teacher_score, tie-break by lower evidence_use_rate.
    best = None
    for cand in sweep_results:
        if cand.get("avg_teacher_score") is None:
            continue
        if best is None:
            best = cand
            continue
        if float(cand["avg_teacher_score"]) > float(best["avg_teacher_score"]) + 1e-9:
            best = cand
            continue
        if abs(float(cand["avg_teacher_score"]) - float(best["avg_teacher_score"])) <= 1e-9:
            cr = cand.get("evidence_use_rate")
            br = best.get("evidence_use_rate")
            if cr is not None and (br is None or float(cr) < float(br)):
                best = cand

    # Optional: validate by running the real gated phase once.
    validate = _truthy(os.getenv("EVOSHOT_GATE_SWEEP_VALIDATE"))
    rows_gated: List[Row] = []
    gated_summary = None
    if validate and best is not None:
        os.environ["EVOSHOT_EVIDENCE_ENABLED"] = "1"
        os.environ["EVOSHOT_EVIDENCE_GATE"] = "fail_or_low_sim"
        os.environ["EVOSHOT_EVIDENCE_MIN_SIM"] = str(best["min_sim"])
        rows_gated = run_images(
            pipeline=pipeline,
            phase=f"sim_topk_with_rules_with_evidence_fail_or_low_sim_{best['min_sim']}",
            prefix="test",
            image_paths=test_imgs,
            post_text=post_text,
        )
        gated_summary = summarize(rows_gated)
        print(f"[VALIDATE gated] min_sim={best['min_sim']} summary={gated_summary}")

    out_dir = Path(os.getenv("EVOSHOT_BENCH_OUT_DIR", str(root / "bench_out")))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"evidence_gate_sweep_cross_{ts}.json"
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
                "summaries": {
                    "train_updates_on": train_summary,
                    "base_no_evidence": base_summary,
                    "evidence_always": ev_summary,
                    "validate_gated": gated_summary,
                },
                "sweep_grid": grid,
                "sweep_results": sweep_results,
                "best": best,
                "validate_enabled": validate,
                "rows_base": [asdict(r) for r in rows_base],
                "rows_evidence_always": [asdict(r) for r in rows_ev],
                "rows_validate_gated": [asdict(r) for r in rows_gated],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results_path": str(out_json),
                "base_summary": base_summary,
                "evidence_summary": ev_summary,
                "best": best,
                "sweep_results_sorted": sweep_results_sorted,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
