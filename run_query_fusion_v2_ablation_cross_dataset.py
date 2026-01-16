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
    retrieval_pool_size: Optional[int]
    fusion_triggered: Optional[bool]
    base_max_sim: Optional[float]
    fused_max_sim: Optional[float]
    teacher_score: Optional[float]
    pass_ge_0_8: Optional[bool]
    teacher_should_add: Optional[bool]
    evidence_facts: int
    seconds: float
    error: Optional[str] = None


def summarize(rows: List[Row]) -> dict:
    scores = [r.teacher_score for r in rows if isinstance(r.teacher_score, (int, float))]
    pass_scores = [s for s in scores if s is not None and s >= 0.8]
    add_flags = [r.teacher_should_add for r in rows if r.teacher_should_add is not None]
    sims = [r.retrieval_max_sim for r in rows if isinstance(r.retrieval_max_sim, (int, float))]
    pools = [r.retrieval_pool_size for r in rows if isinstance(r.retrieval_pool_size, int)]
    facts = [r.evidence_facts for r in rows if isinstance(r.evidence_facts, int)]
    used = [1 for r in rows if isinstance(r.evidence_facts, int) and r.evidence_facts > 0]
    secs = [r.seconds for r in rows if isinstance(r.seconds, (int, float))]
    fused = [1 for r in rows if r.fusion_triggered is True]
    base_sims = [r.base_max_sim for r in rows if isinstance(r.base_max_sim, (int, float))]
    fused_sims = [r.fused_max_sim for r in rows if isinstance(r.fused_max_sim, (int, float))]

    def avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": len(rows),
        "avg_teacher_score": avg([float(x) for x in scores if x is not None]),
        "pass_rate_ge_0_8": (len(pass_scores) / len(scores)) if scores else None,
        "add_rate_should_add": (sum(1 for x in add_flags if x) / len(add_flags)) if add_flags else None,
        "avg_retrieval_max_sim": avg([float(x) for x in sims if x is not None]),
        "avg_retrieval_pool_size": avg([float(x) for x in pools if x is not None]),
        "avg_base_max_sim": avg([float(x) for x in base_sims if x is not None]),
        "avg_fused_max_sim": avg([float(x) for x in fused_sims if x is not None]),
        "fusion_trigger_rate": (sum(fused) / len(rows)) if rows else None,
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
        pool_size = meta.get("pool_size")
        try:
            max_sim_f = float(max_sim) if max_sim is not None else None
        except Exception:
            max_sim_f = None
        try:
            pool_size_i = int(pool_size) if pool_size is not None else None
        except Exception:
            pool_size_i = None

        base_meta = trace.get("retrieval_meta_base") or {}
        fused_meta = trace.get("retrieval_meta_fused") or {}
        try:
            base_sim_f = float(base_meta.get("max_sim")) if base_meta.get("max_sim") is not None else None
        except Exception:
            base_sim_f = None
        try:
            fused_sim_f = float(fused_meta.get("max_sim")) if fused_meta.get("max_sim") is not None else None
        except Exception:
            fused_sim_f = None

        fusion_triggered = trace.get("retrieval_fusion_triggered")
        fusion_triggered_b = bool(fusion_triggered) if fusion_triggered is not None else None

        facts = trace.get("evidence_facts") or []
        n_facts = len(facts) if isinstance(facts, list) else 0

        score = trace.get("teacher_score")
        score_f = float(score) if isinstance(score, (int, float)) else None

        should_add = trace.get("teacher_should_add")
        should_add_b = bool(should_add) if should_add is not None else None

        out.append(
            Row(
                phase=phase,
                sample_id=sample_id,
                image_path=str(image_path),
                retrieval_max_sim=max_sim_f,
                retrieval_pool_size=pool_size_i,
                fusion_triggered=fusion_triggered_b,
                base_max_sim=base_sim_f,
                fused_max_sim=fused_sim_f,
                teacher_score=score_f,
                pass_ge_0_8=(score_f >= 0.8) if score_f is not None else None,
                teacher_should_add=should_add_b,
                evidence_facts=n_facts,
                seconds=dt,
                error=str(trace.get("error")) if trace.get("error") else None,
            )
        )
    return out


def main() -> int:
    root = Path(__file__).resolve().parent
    _load_dotenv(root / ".env", override=False)

    # Backends
    os.environ.setdefault("EVOSHOT_STUDENT_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_EMBED_BACKEND", "real")
    os.environ.setdefault("EVOSHOT_TEACHER_BACKEND", "real")

    # Retrieval/query defaults
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

    # Baseline evidence gate (best-known on cross-dataset sweep)
    os.environ.setdefault("EVOSHOT_RULEBANK_ENABLED", "1")
    os.environ.setdefault("EVOSHOT_RULES_K", "4")
    os.environ.setdefault("EVOSHOT_EVIDENCE_ENABLED", "1")
    os.environ.setdefault("EVOSHOT_EVIDENCE_GATE", "fail_or_low_sim")
    os.environ.setdefault("EVOSHOT_EVIDENCE_MIN_SIM", "0.68")

    train_n = int(os.getenv("EVOSHOT_TRAIN_N", "8"))
    test_n = int(os.getenv("EVOSHOT_TEST_N", "16"))
    batch_seed = int(os.getenv("EVOSHOT_BATCH_SEED", "43"))
    post_text = os.getenv("EVOSHOT_DEFAULT_POST_TEXT", "Please score this image.")
    retrieval_k = int(os.getenv("EVOSHOT_RETRIEVAL_K", "2"))
    retrieval_k = max(0, retrieval_k)
    rules_k = int(os.getenv("EVOSHOT_RULES_K", "4"))
    rules_k = max(0, rules_k)
    min_sim = float(os.getenv("EVOSHOT_EVIDENCE_MIN_SIM", "0.68"))

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
    out_json = out_dir / f"query_fusion_v2_ablation_cross_{ts}.json"

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

    phases = [
        (
            "baseline_best",
            {
                "EVOSHOT_RULEBANK_ENABLED": "1",
                "EVOSHOT_RULES_K": str(rules_k),
                "EVOSHOT_EVIDENCE_ENABLED": "1",
                "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
                "EVOSHOT_EVIDENCE_MIN_SIM": str(min_sim),
                "EVOSHOT_RETRIEVAL_CAPTION_STYLE": "sentence",
                "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION": "0",
            },
        ),
        (
            "union_rerank_max_always",
            {
                "EVOSHOT_RULEBANK_ENABLED": "1",
                "EVOSHOT_RULES_K": str(rules_k),
                "EVOSHOT_EVIDENCE_ENABLED": "1",
                "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
                "EVOSHOT_EVIDENCE_MIN_SIM": str(min_sim),
                "EVOSHOT_RETRIEVAL_CAPTION_STYLE": "sentence",
                "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION": "1",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_GATE": "always",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_MODE": "max",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_SET": "union_topm",
            },
        ),
        (
            "union_rerank_weighted_always",
            {
                "EVOSHOT_RULEBANK_ENABLED": "1",
                "EVOSHOT_RULES_K": str(rules_k),
                "EVOSHOT_EVIDENCE_ENABLED": "1",
                "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
                "EVOSHOT_EVIDENCE_MIN_SIM": str(min_sim),
                "EVOSHOT_RETRIEVAL_CAPTION_STYLE": "sentence",
                "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION": "1",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_GATE": "always",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_MODE": "weighted",
                "EVOSHOT_RETRIEVAL_QUERY_WEIGHTS": "0.25,0.25,0.5",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_SET": "union_topm",
            },
        ),
        (
            "union_rerank_weighted_gate_fail_or_low_sim",
            {
                "EVOSHOT_RULEBANK_ENABLED": "1",
                "EVOSHOT_RULES_K": str(rules_k),
                "EVOSHOT_EVIDENCE_ENABLED": "1",
                "EVOSHOT_EVIDENCE_GATE": "fail_or_low_sim",
                "EVOSHOT_EVIDENCE_MIN_SIM": str(min_sim),
                "EVOSHOT_RETRIEVAL_CAPTION_STYLE": "sentence",
                "EVOSHOT_RETRIEVAL_QUERY_EXPANSION": "0",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION": "1",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_GATE": "fail_or_low_sim",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_MIN_SIM": str(min_sim),
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_MODE": "weighted",
                "EVOSHOT_RETRIEVAL_QUERY_WEIGHTS": "0.25,0.25,0.5",
                "EVOSHOT_RETRIEVAL_QUERY_FUSION_SET": "union_topm",
            },
        ),
    ]

    summaries: dict[str, object] = {"train_updates_on": train_summary}
    rows_all: List[Row] = []
    rows_all.extend(rows_train)
    completed_phases: List[str] = ["train_updates_on"]

    def _checkpoint() -> None:
        out_json.write_text(
            json.dumps(
                {
                    "train_n": train_n,
                    "test_n": test_n,
                    "batch_seed": batch_seed,
                    "retrieval_k": retrieval_k,
                    "rules_k": rules_k,
                    "evidence_gate_min_sim": min_sim,
                    "filter_stats": filter_stats,
                    "train_images": [str(p) for p in train_imgs],
                    "test_images": [str(p) for p in test_imgs],
                    "summaries": summaries,
                    "rows": [asdict(r) for r in rows_all],
                    "phases": [{"name": name, "env": env} for name, env in phases],
                    "completed_phases": list(completed_phases),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    _checkpoint()

    for phase, env in phases:
        for k, v in env.items():
            os.environ[k] = v
        rows = run_images(pipeline=pipeline, phase=phase, prefix="test", image_paths=test_imgs, post_text=post_text)
        summaries[phase] = summarize(rows)
        rows_all.extend(rows)
        completed_phases.append(phase)
        _checkpoint()
        print(f"[EVAL] {phase}: {summaries[phase]}")

    print(json.dumps({"results_path": str(out_json), "summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

