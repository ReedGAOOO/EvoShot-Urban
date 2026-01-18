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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


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
    split: str
    sample_id: str
    image_path: str
    teacher_score: Optional[float]
    teacher_should_add: Optional[bool]
    retrieval_max_sim: Optional[float]
    retrieved_ids: List[str]
    rules_used: int
    rulebank_injected: Optional[bool]
    evidence_facts: int
    shots_dropped: Optional[bool]
    self_revision_triggered: Optional[bool]
    seconds: Optional[float] = None
    error: Optional[str] = None


def _avg(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _rate_true(xs: List[Optional[bool]]) -> Optional[float]:
    vals = [bool(x) for x in xs if x is not None]
    return (sum(1 for x in vals if x) / len(vals)) if vals else None


def summarize(rows: List[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    add_flags = [r.teacher_should_add for r in rows if r.teacher_should_add is not None]
    max_sims = [r.retrieval_max_sim for r in rows if isinstance(r.retrieval_max_sim, (int, float))]

    evidence_use = [r.evidence_facts > 0 for r in rows]
    rule_injected = [r.rulebank_injected for r in rows]
    shot_dropped = [r.shots_dropped for r in rows]
    self_rev = [r.self_revision_triggered for r in rows]

    return {
        "n": len(rows),
        "avg_teacher_score": _avg([float(x) for x in scores if x is not None]),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "add_rate_should_add": (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None,
        "avg_retrieval_max_sim": _avg([float(x) for x in max_sims if x is not None]),
        "evidence_use_rate": (sum(1 for x in evidence_use if x) / len(evidence_use)) if evidence_use else None,
        "rulebank_injected_rate": _rate_true(rule_injected),
        "shots_dropped_rate": _rate_true(shot_dropped),
        "self_revision_rate": _rate_true(self_rev),
        "avg_seconds": _avg([float(r.seconds) for r in rows if isinstance(r.seconds, (int, float))]),
    }


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


def _aggregate(seed_payloads: List[dict], phase: str, split: str) -> dict:
    vals = []
    for s in seed_payloads:
        v = ((s.get("summaries_by_phase") or {}).get(phase) or {}).get(split, {}).get("avg_teacher_score")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return {
        "phase": phase,
        "split": split,
        "n_seeds": len(seed_payloads),
        "avg_teacher_score_mean": mean(vals) if vals else None,
        "avg_teacher_score_stdev": pstdev(vals) if len(vals) > 1 else None,
    }


def _set_env(cfg: dict) -> None:
    for k, v in cfg.items():
        os.environ[k] = str(v)


def run_images(
    *,
    seed: int,
    pipeline: UrbanPipeline,
    phase: str,
    split: str,
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
        dt = time.time() - t0

        meta = trace.get("retrieval_meta") or {}
        max_sim = meta.get("max_sim")
        try:
            max_sim_f = float(max_sim) if max_sim is not None else None
        except Exception:
            max_sim_f = None

        facts = trace.get("evidence_facts") or []
        rule_ids = trace.get("rule_ids") or []
        out.append(
            Row(
                seed=seed,
                phase=phase,
                split=split,
                sample_id=sample_id,
                image_path=str(image_path),
                teacher_score=float(trace["teacher_score"])
                if isinstance(trace.get("teacher_score"), (int, float))
                else None,
                teacher_should_add=bool(trace["teacher_should_add"])
                if trace.get("teacher_should_add") is not None
                else None,
                retrieval_max_sim=max_sim_f,
                retrieved_ids=list(trace.get("retrieved_ids") or []),
                rules_used=len(rule_ids) if isinstance(rule_ids, list) else 0,
                rulebank_injected=bool(trace.get("rulebank_injected"))
                if trace.get("rulebank_injected") is not None
                else None,
                evidence_facts=len(facts) if isinstance(facts, list) else 0,
                shots_dropped=bool(trace.get("shots_dropped"))
                if trace.get("shots_dropped") is not None
                else None,
                self_revision_triggered=bool(trace.get("self_revision_triggered"))
                if trace.get("self_revision_triggered") is not None
                else None,
                seconds=float(dt),
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def _select_train_images(*, train_root: Path, train_n: int, batch_seed: int) -> tuple[List[Path], dict]:
    pool_factor = int(os.getenv("EVOSHOT_FILTER_POOL_FACTOR", "8"))
    min_conf = float(os.getenv("EVOSHOT_FILTER_MIN_CONF", "0.55"))
    max_factor = int(os.getenv("EVOSHOT_FILTER_MAX_FACTOR", "40"))

    if train_n <= 0:
        return [], {"train_n": train_n, "selected": 0, "pool_factor": 0, "min_conf": min_conf}

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
    stats = {
        "train_n": train_n,
        "selected": len(train_imgs),
        "pool_factor_start": pool_factor,
        "pool_factor_used": used_factor,
        "attempts": attempts,
        "min_conf": min_conf,
    }
    print(f"[FILTER] Done: selected {len(train_imgs)}/{train_n} | stats={stats}")
    return train_imgs, stats


def _select_test_images(*, test_roots: List[Path], test_n: int, batch_seed: int) -> List[Path]:
    per_root = max(1, (test_n + len(test_roots) - 1) // len(test_roots)) if test_n > 0 else 0
    test_imgs: List[Path] = []
    for i, r in enumerate(test_roots):
        if per_root <= 0:
            continue
        test_imgs.extend(reservoir_sample_images(r, k=per_root, recurse=False, seed=batch_seed + 100 + i))
    return test_imgs[:test_n]


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
    os.environ.setdefault("EVOSHOT_RETRIEVAL_STRATEGY", "topk")
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

    train_n = int(os.getenv("EVOSHOT_TRAIN_N", "8"))
    test_n = int(os.getenv("EVOSHOT_TEST_N", "16"))
    batch_seed = int(os.getenv("EVOSHOT_BATCH_SEED", "42"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")
    seeds = _parse_seeds(os.getenv("EVOSHOT_EVAL_SEEDS", "42,43,44"))

    train_root = Path(os.getenv("EVOSHOT_DATASET1_ROOT", r"H:\微博数据 - 副本\images"))
    test_roots = [
        Path(os.getenv("EVOSHOT_CITYSCAPES_HANOVER", r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\hanover")),
        Path(os.getenv("EVOSHOT_CITYSCAPES_AACHEN", r"H:\leftImg8bit_trainvaltest\leftImg8bit\train\aachen")),
    ]
    if not train_root.exists():
        raise FileNotFoundError(f"Train root not found: {train_root}")
    for r in test_roots:
        if not r.exists():
            raise FileNotFoundError(f"Test root not found: {r}")

    train_imgs, filter_stats = _select_train_images(train_root=train_root, train_n=train_n, batch_seed=batch_seed)
    test_imgs = _select_test_images(test_roots=test_roots, test_n=test_n, batch_seed=batch_seed)
    print(f"[DATA] Fixed test set: {len(test_imgs)}/{test_n} images")

    base_train_cfg = {
        "EVOSHOT_DISABLE_VAULT_UPDATE": "0",
        "EVOSHOT_RULEBANK_ENABLED": "0",
        "EVOSHOT_EVIDENCE_ENABLED": "0",
        "EVOSHOT_SHOTS_FORMAT": "full",
        "EVOSHOT_SHOTS_MIN_SIM": "",
        "EVOSHOT_STUDENT_VISUAL_EVIDENCE": "0",
        "EVOSHOT_SELF_REVISION_ENABLED": "0",
        "EVOSHOT_RULEBANK_GATE": "always",
        "EVOSHOT_VAULT_REQUIRE_LESSON": "0",
    }

    base_eval_cfg = {
        "EVOSHOT_DISABLE_VAULT_UPDATE": "1",
        "EVOSHOT_RETRIEVAL_STRATEGY": "topk",
        "EVOSHOT_RETRIEVAL_K": "2",
        "EVOSHOT_RULEBANK_ENABLED": "1",
        "EVOSHOT_RULES_K": "4",
        "EVOSHOT_EVIDENCE_ENABLED": "1",
        "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
        "EVOSHOT_EVIDENCE_MIN_SIM": "0.68",
        "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
        "EVOSHOT_SHOTS_FORMAT": "full",
        "EVOSHOT_SHOTS_MIN_SIM": "",
        "EVOSHOT_STUDENT_VISUAL_EVIDENCE": "0",
        "EVOSHOT_SELF_REVISION_ENABLED": "0",
        "EVOSHOT_RULEBANK_GATE": "always",
        "EVOSHOT_RULEBANK_MIN_SIM": "0.68",
    }

    phases: List[tuple[str, dict]] = [
        ("baseline_trunk", {}),
        ("shot_min_sim_0.68", {"EVOSHOT_SHOTS_MIN_SIM": "0.68"}),
        ("shots_grounding_first", {"EVOSHOT_SHOTS_FORMAT": "grounding_first"}),
        ("student_visual_evidence", {"EVOSHOT_STUDENT_VISUAL_EVIDENCE": "1"}),
        ("self_revision_low_sim_or_low_conf", {"EVOSHOT_SELF_REVISION_ENABLED": "1"}),
        ("rulebank_gate_fail_or_low_sim", {"EVOSHOT_RULEBANK_GATE": "fail_or_low_sim"}),
        (
            "combo_all",
            {
                "EVOSHOT_SHOTS_MIN_SIM": "0.68",
                "EVOSHOT_SHOTS_FORMAT": "grounding_first",
                "EVOSHOT_STUDENT_VISUAL_EVIDENCE": "1",
                "EVOSHOT_SELF_REVISION_ENABLED": "1",
                "EVOSHOT_RULEBANK_GATE": "fail_or_low_sim",
            },
        ),
    ]

    include_strict_vault = (os.getenv("EVOSHOT_INCLUDE_STRICT_VAULT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    all_rows: List[Row] = []
    seed_payloads: List[dict] = []

    out_dir = root / "bench_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"icling_improvements_ablation_cross_{ts}.json"
    checkpoint_path = out_dir / f"icling_improvements_ablation_cross_{ts}.checkpoint.json"

    for s in seeds:
        print(f"\n=== Seed {s} ===")
        os.environ["EVOSHOT_RANDOM_SEED"] = str(s)

        # -------------------------
        # Train (updates ON): build Vault + RuleBank
        # -------------------------
        _set_env(base_train_cfg)
        _set_env({"EVOSHOT_VAULT_REQUIRE_LESSON": "0"})
        pipeline = UrbanPipeline()
        rows_train = run_images(
            seed=s,
            pipeline=pipeline,
            phase="train_updates_on",
            split="train",
            prefix=f"train_s{s}",
            image_paths=train_imgs,
            post_text=post_text,
        )

        # Freeze updates
        os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"

        summaries_by_phase: dict[str, dict] = {
            "train_updates_on": {"train": summarize(rows_train)},
        }
        all_rows.extend(rows_train)

        # -------------------------
        # Evaluate inference-time ablations on SAME trained pipeline
        # -------------------------
        for phase_name, overrides in phases:
            cfg = dict(base_eval_cfg)
            cfg.update(overrides)
            _set_env(cfg)
            rows_test = run_images(
                seed=s,
                pipeline=pipeline,
                phase=phase_name,
                split="test",
                prefix=f"test_{phase_name}_s{s}",
                image_paths=test_imgs,
                post_text=post_text,
            )
            summaries_by_phase.setdefault(phase_name, {})["test"] = summarize(rows_test)
            all_rows.extend(rows_test)

        # -------------------------
        # Optional: strict vault add gating (training-time)
        # -------------------------
        strict_summary = None
        if include_strict_vault:
            _set_env(base_train_cfg)
            _set_env({"EVOSHOT_VAULT_REQUIRE_LESSON": "1"})
            os.environ["EVOSHOT_RANDOM_SEED"] = str(s)
            pipeline_strict = UrbanPipeline()
            rows_train_strict = run_images(
                seed=s,
                pipeline=pipeline_strict,
                phase="train_updates_on_strict_vault",
                split="train",
                prefix=f"train_strict_s{s}",
                image_paths=train_imgs,
                post_text=post_text,
            )
            os.environ["EVOSHOT_DISABLE_VAULT_UPDATE"] = "1"
            _set_env(base_eval_cfg)
            rows_test_strict = run_images(
                seed=s,
                pipeline=pipeline_strict,
                phase="baseline_trunk_strict_vault",
                split="test",
                prefix=f"test_baseline_strict_s{s}",
                image_paths=test_imgs,
                post_text=post_text,
            )
            strict_summary = {
                "train": summarize(rows_train_strict),
                "test": summarize(rows_test_strict),
            }
            all_rows.extend(rows_train_strict)
            all_rows.extend(rows_test_strict)

        seed_payloads.append(
            {
                "seed": s,
                "train_summary": summarize(rows_train),
                "summaries_by_phase": summaries_by_phase,
                "strict_vault": strict_summary,
            }
        )

        # Checkpoint after each seed to avoid losing progress on long runs.
        try:
            checkpoint_payload = {
                "train_n": train_n,
                "test_n": test_n,
                "batch_seed": batch_seed,
                "seeds": list(seeds),
                "completed_seeds": [p.get("seed") for p in seed_payloads],
                "filter_stats": filter_stats,
                "phases": [p for p, _ in phases],
                "include_strict_vault": bool(include_strict_vault),
                "seed_payloads": seed_payloads,
                "rows_n": len(all_rows),
                "last_updated_ts": int(time.time()),
            }
            checkpoint_path.write_text(
                json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

    payload = {
        "train_n": train_n,
        "test_n": test_n,
        "batch_seed": batch_seed,
        "seeds": list(seeds),
        "filter_stats": filter_stats,
        "train_images": [str(p) for p in train_imgs],
        "test_images": [str(p) for p in test_imgs],
        "phases": [p for p, _ in phases],
        "include_strict_vault": bool(include_strict_vault),
        "seed_payloads": seed_payloads,
        "aggregate": {
            "baseline_trunk_test": _aggregate(seed_payloads, "baseline_trunk", "test"),
            "shot_min_sim_0.68_test": _aggregate(seed_payloads, "shot_min_sim_0.68", "test"),
            "shots_grounding_first_test": _aggregate(seed_payloads, "shots_grounding_first", "test"),
            "student_visual_evidence_test": _aggregate(seed_payloads, "student_visual_evidence", "test"),
            "self_revision_low_sim_or_low_conf_test": _aggregate(seed_payloads, "self_revision_low_sim_or_low_conf", "test"),
            "rulebank_gate_fail_or_low_sim_test": _aggregate(seed_payloads, "rulebank_gate_fail_or_low_sim", "test"),
            "combo_all_test": _aggregate(seed_payloads, "combo_all", "test"),
        },
        "rows": [asdict(r) for r in all_rows],
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
