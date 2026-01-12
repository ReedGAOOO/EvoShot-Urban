import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _bin_score(s: Optional[float]) -> str:
    if s is None:
        return "na"
    if s < 0.2:
        return "[0.0,0.2)"
    if s < 0.4:
        return "[0.2,0.4)"
    if s < 0.6:
        return "[0.4,0.6)"
    if s < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def tag_critique(text: str) -> List[str]:
    t = (text or "").lower()
    if not t.strip():
        return []

    tags: List[str] = []

    def has(*subs: str) -> bool:
        return any(s in t for s in subs)

    # Domain mismatch / scene type issues
    if has(
        "not an urban",
        "non-urban",
        "forest",
        "beach",
        "mountain",
        "portrait",
        "wild",
        "park",
        "nature",
        "landscape",
        "interior",
        "indoor",
        "private",
        "retail",
        "restaurant",
        "food",
        "clothing",
    ):
        tags.append("domain_mismatch")

    # Hallucination / grounding
    if has(
        "hallucinat",
        "not visible",
        "unseen",
        "made up",
        "fabricat",
        "invent",
        "vague",
        "generic",
        "lacks specific",
        "lacks specificity",
        "poor grounding",
        "fails to provide specific",
    ):
        tags.append("grounding_or_hallucination")

    # Missed key details / omissions
    if has("missed", "failed to", "ignored", "didn't", "does not adequately", "does not sufficiently"):
        tags.append("missed_details")

    # Lighting / time confusion
    if has("daylight", "overcast", "night", "dim", "street lamp", "lighting", "glare", "visibility"):
        tags.append("lighting_or_visibility")

    # Common object-level failure themes
    if has("pedestrian", "crowd", "foot traffic", "people"):
        tags.append("people_activity")
    if has("traffic", "car", "vehicle", "parked", "van", "bus", "bike"):
        tags.append("vehicles_traffic")
    if has("construction", "barrier", "obstruction", "blocked", "hazard", "wire"):
        tags.append("construction_or_hazards")
    if has("trash", "litter", "overflow", "bin"):
        tags.append("trash_cleanliness")
    if has("signage", "text on", "failed to identify", "clear text", "read"):
        tags.append("missed_text_signage")

    # Score-dimension mentions
    if "safety" in t:
        tags.append("safety")
    if "vibrancy" in t:
        tags.append("vibrancy")
    if "cleanliness" in t:
        tags.append("cleanliness")

    # Deduplicate while preserving order
    seen = set()
    out = []
    for tag in tags:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out


@dataclass(frozen=True)
class RowKey:
    dataset: str
    mode: str
    image_path: str


def summarize(results: List[dict]) -> dict:
    scores = [_safe_float(r.get("teacher_score")) for r in results]
    vals = [s for s in scores if s is not None]
    pass_rate = None
    if vals:
        pass_rate = sum(1 for s in vals if s >= 0.8) / len(vals)
    tag_counts = Counter()
    for r in results:
        for tag in tag_critique(r.get("critique", "")):
            tag_counts[tag] += 1

    bins = Counter(_bin_score(s) for s in scores)
    return {
        "n": len(results),
        "avg": mean(vals) if vals else None,
        "median": median(vals) if vals else None,
        "stdev": pstdev(vals) if len(vals) > 1 else None,
        "pass_rate_ge_0_8": pass_rate,
        "score_bins": bins,
        "top_tags": tag_counts.most_common(12),
    }


def group_by(results: List[dict]) -> Dict[Tuple[str, str], List[dict]]:
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in results:
        groups[(r.get("dataset", ""), r.get("mode", ""))].append(r)
    return groups


def build_map(results: List[dict]) -> Dict[RowKey, dict]:
    m: Dict[RowKey, dict] = {}
    for r in results:
        m[RowKey(r.get("dataset", ""), r.get("mode", ""), r.get("image_path", ""))] = r
    return m


def compare(old_results: List[dict], new_results: List[dict]) -> dict:
    old_map = build_map(old_results)
    new_map = build_map(new_results)
    keys = sorted(set(old_map) & set(new_map), key=lambda k: (k.dataset, k.mode, k.image_path))

    deltas = []
    flips = {"old_pass_new_fail": 0, "old_fail_new_pass": 0}
    both_halluc = 0
    total = 0

    for k in keys:
        o = _safe_float(old_map[k].get("teacher_score"))
        n = _safe_float(new_map[k].get("teacher_score"))
        if o is None or n is None:
            continue
        total += 1
        deltas.append(n - o)
        if (o >= 0.8) and (n < 0.8):
            flips["old_pass_new_fail"] += 1
        if (o < 0.8) and (n >= 0.8):
            flips["old_fail_new_pass"] += 1

        o_tags = set(tag_critique(old_map[k].get("critique", "")))
        n_tags = set(tag_critique(new_map[k].get("critique", "")))
        if ("grounding_or_hallucination" in o_tags) and ("grounding_or_hallucination" in n_tags):
            both_halluc += 1

    return {
        "paired_n": total,
        "delta_avg_new_minus_old": mean(deltas) if deltas else None,
        "delta_median_new_minus_old": median(deltas) if deltas else None,
        "flips": flips,
        "both_grounding_or_hallucination_rate": (both_halluc / total) if total else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="First results JSON (baseline)")
    ap.add_argument("--b", required=True, help="Second results JSON (comparison)")
    args = ap.parse_args()

    a_path = Path(args.a).resolve()
    b_path = Path(args.b).resolve()
    a = load_results(a_path)
    b = load_results(b_path)

    a_res = a.get("results") or []
    b_res = b.get("results") or []

    out = {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_summary_overall": summarize(a_res),
        "b_summary_overall": summarize(b_res),
        "a_by_group": {f"{k[0]}::{k[1]}": summarize(v) for k, v in group_by(a_res).items()},
        "b_by_group": {f"{k[0]}::{k[1]}": summarize(v) for k, v in group_by(b_res).items()},
        "paired_compare": compare(a_res, b_res),
    }

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

