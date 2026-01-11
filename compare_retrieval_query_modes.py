import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from urban_experiment import Sample, UrbanPipeline


@dataclass
class RoundResult:
    mode: str
    image_name: str
    r1_teacher_score: Optional[float]
    r2_teacher_score: Optional[float]
    r1_added: Optional[bool]
    r2_added: Optional[bool]
    r2_hit_r1_in_retrieval: Optional[bool]
    r1_retrieved: List[str]
    r2_retrieved: List[str]
    notes: str = ""


def _dummy_gt() -> dict:
    return {"safety": 3.0, "vibrancy": 3.0, "cleanliness": 3.0}


def _pick_images(pic_dir: Path, n: int) -> List[Path]:
    imgs = sorted(list(pic_dir.glob("*.png")) + list(pic_dir.glob("*.jpg")) + list(pic_dir.glob("*.jpeg")))
    return imgs[:n]


def _run_two_rounds_for_image(image_path: Path, *, mode: str, post_text: str) -> RoundResult:
    os.environ["EVOSHOT_QUERY_EMBED_MODE"] = mode

    pipeline = UrbanPipeline()

    s1 = Sample(
        id=f"{image_path.stem}_r1",
        image_path=str(image_path),
        post_text=post_text,
        ground_truth_sim=_dummy_gt(),
    )
    t1 = pipeline.process_sample(s1)

    s2 = Sample(
        id=f"{image_path.stem}_r2",
        image_path=str(image_path),
        post_text=post_text,
        ground_truth_sim=_dummy_gt(),
    )
    t2 = pipeline.process_sample(s2)

    r1_score = t1.get("teacher_score")
    r2_score = t2.get("teacher_score")
    r1_added = t1.get("teacher_should_add")
    r2_added = t2.get("teacher_should_add")
    r1_id = t1.get("vault_added_id")
    r2_retrieved = t2.get("retrieved_ids") or []
    hit = None
    if r1_id is not None:
        hit = r1_id in r2_retrieved

    return RoundResult(
        mode=mode,
        image_name=image_path.name,
        r1_teacher_score=r1_score,
        r2_teacher_score=r2_score,
        r1_added=r1_added,
        r2_added=r2_added,
        r2_hit_r1_in_retrieval=hit,
        r1_retrieved=t1.get("retrieved_ids") or [],
        r2_retrieved=r2_retrieved,
        notes=("r1_error=" + t1.get("error")) if t1.get("error") else "",
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    pic_dir = Path(os.getenv("EVOSHOT_PIC_DIR", str(root / "PIC_DATA"))).resolve()
    n = int(os.getenv("EVOSHOT_EXPERIMENT_N", "3"))
    post_text = os.getenv("EVOSHOT_EXPERIMENT_POST_TEXT", "Please score this urban street scene.")

    images = _pick_images(pic_dir, n=n)
    if not images:
        raise FileNotFoundError(f"No images found in {pic_dir}")

    modes = ["text", "caption+text"]
    results: List[RoundResult] = []

    for img in images:
        for mode in modes:
            results.append(_run_two_rounds_for_image(img, mode=mode, post_text=post_text))

    print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))

    def _avg(xs: List[Optional[float]]) -> Optional[float]:
        vals = [x for x in xs if isinstance(x, (int, float))]
        return sum(vals) / len(vals) if vals else None

    for mode in modes:
        sub = [r for r in results if r.mode == mode]
        avg1 = _avg([r.r1_teacher_score for r in sub])
        avg2 = _avg([r.r2_teacher_score for r in sub])
        hit_rate = None
        hits = [r.r2_hit_r1_in_retrieval for r in sub if r.r2_hit_r1_in_retrieval is not None]
        if hits:
            hit_rate = sum(1 for h in hits if h) / len(hits)
        print(
            f"\nSUMMARY mode={mode}: avg_r1={avg1} avg_r2={avg2} r2_hit_r1_rate={hit_rate} (n={len(sub)})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

