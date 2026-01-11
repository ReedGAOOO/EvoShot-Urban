import argparse
import os
import time
from pathlib import Path

from urban_experiment import Sample, UrbanPipeline


def _dummy_gt() -> dict:
    return {"safety": 3.0, "vibrancy": 3.0, "cleanliness": 3.0}


def run_round(pipeline: UrbanPipeline, *, sample_id: str, image_path: Path, post_text: str) -> None:
    sample = Sample(
        id=sample_id,
        image_path=str(image_path),
        post_text=post_text,
        ground_truth_sim=_dummy_gt(),
    )
    pipeline.process_sample(sample)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train image")
    ap.add_argument("--test", required=False, help="Path to test image (optional)")
    ap.add_argument("--train-text", default="Analyze this urban scene.", help="Post text for train sample")
    ap.add_argument("--test-text", default="Analyze this urban scene.", help="Post text for test sample")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between rounds (seconds)")
    args = ap.parse_args()

    train_path = Path(args.train).expanduser().resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Train image not found: {train_path}")

    test_path = None
    if args.test:
        test_path = Path(args.test).expanduser().resolve()
        if not test_path.exists():
            raise FileNotFoundError(f"Test image not found: {test_path}")

    print("[ENV] EVOSHOT_STUDENT_BACKEND =", os.getenv("EVOSHOT_STUDENT_BACKEND"))
    print("[ENV] EVOSHOT_EMBED_BACKEND   =", os.getenv("EVOSHOT_EMBED_BACKEND"))
    print("[ENV] EVOSHOT_TEACHER_BACKEND =", os.getenv("EVOSHOT_TEACHER_BACKEND"))
    print("[ENV] EVOSHOT_RETRIEVAL_STRATEGY =", os.getenv("EVOSHOT_RETRIEVAL_STRATEGY"))

    pipeline = UrbanPipeline()

    print("\n=== Round 1 (train) ===")
    run_round(pipeline, sample_id="train_round1", image_path=train_path, post_text=args.train_text)
    time.sleep(args.sleep)

    print("\n=== Vault Summary (after Round 1) ===")
    pipeline.vault.summarize()

    print("\n=== Round 2 (train again) ===")
    run_round(pipeline, sample_id="train_round2", image_path=train_path, post_text=args.train_text)
    time.sleep(args.sleep)

    if test_path:
        print("\n=== Round 3 (test) ===")
        run_round(pipeline, sample_id="test_round3", image_path=test_path, post_text=args.test_text)

    print("\n=== Final Vault Summary ===")
    pipeline.vault.summarize()
    print("\nFinal Vault Size:", len(pipeline.vault.storage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

