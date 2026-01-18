# EvoShot-Urban

EvoShot (Evolutionary Few-Shot System): a Teacher-Student, retrieval-augmented in-context learning framework (**no fine-tuning**).

This repository ships the first case study: **urban environment perception** from images (to demonstrate cost-effective cloud-teacher guidance for a local mid/small student model; see the cost sketch below).

Chinese README: `README.zh-CN.md`.

## EvoShot in one paragraph

EvoShot is a non-parametric learning loop: a cheap/local **Student** handles most inputs; a stronger (cloud) **Teacher** grades and corrects only when needed; the system then stores the correction as an evolving few-shot memory (Vault + RuleBank). Future queries retrieve relevant items and inject them into the Student prompt (ICL only), improving quality while reducing total teacher calls/cost. The loop is task-agnostic: swap the output schema, teacher rubric, and memory formatting to apply it to other domains.

## At a glance (urban case)

Quality (Teacher reward in `[0,1]`, higher is better):

- Student-only (no retrieval / no RuleBank / no Evidence): `avg_teacher_score=0.54`, `pass_rate>=0.8=0%`
- EvoShot best-practice trunk: `avg_teacher_score=0.762` (**+0.222**, ~`+41%` vs Student-only; ~`76%` of full marks), `pass_rate>=0.8=30%`

Source (same protocol, same batch): `bench_out/rulebank_evidence_ablation_cross_20260114_201832.json`.

Stability note: under the cross-dataset 3-batch sweep, the tuned trunk config (`EVOSHOT_EVIDENCE_GATE=fail_or_low_sim`, `EVOSHOT_EVIDENCE_MIN_SIM=0.68`) reached mean `avg_teacher_score~0.7296` (sources below).

Cost (back-of-envelope, set per-item cloud-teacher cost to `x`):

- Teacher-only baseline on `N` images: cost ~ `N * x`
- EvoShot: Student runs locally for all `N`; Teacher can be gated to only intervene on a fraction `p` of inputs (e.g., low-sim / low-confidence / audit). Cost ~ `p * N * x`
  - In our cross-dataset protocol, the "retrieval weak -> trigger scaffolding" rate at `EVOSHOT_EVIDENCE_MIN_SIM=0.68` was ~`0.40`, implying ~`60%` fewer teacher calls if you gate teacher on the same signal (evaluation scripts still call Teacher to score).
  - Example: `N=10,000` images -> Teacher-only ~ `10,000*x`; EvoShot with `p=0.40` ~ `4,000*x` (~`2.5x` cheaper).
  - Source: `bench_out/evidence_gate_sweep_cross_20260115_101213.json`, `bench_out/evidence_gate_sweep_cross_20260115_103644.json`, `bench_out/evidence_gate_sweep_cross_20260115_111504.json`.
  - Practical note: evaluation runners call Teacher to compute `avg_teacher_score`; for cheap deployment inference, run Student-only with retrieval/memory (and reserve Teacher for active learning or audits).

### Case study in this repo: Urban perception scoring

Given an image (+ optional post text), the Student VLM outputs:

- `safety`, `vibrancy`, `cleanliness` (1.0-5.0)

## Architecture (Teacher-Student + Memory)

- **Student** (`models/student.py`): OpenAI-compatible `/v1/chat/completions` VLM + `robust_json_parser()` (handles messy JSON).
- **Teacher** (`models/teacher.py`): strict grader + improved rationale/scores (+ optional caption). Real teacher **costs money**.
- **Vault** (`urban_experiment.py`): stores Tier-0 seeds + Tier-1 teacher-corrected examples; retrieves top-k / MMR.
- **Embeddings** (`embedding_test.py` contract): text embeddings for retrieval via an embedding service (`EVOSHOT_EMBED_BACKEND=real`).
- **Evidence** (`models/evidence.py`): extracts "visible facts" as scaffolding to reduce hallucination.
- **RuleBank** (`models/rules.py`): stores short reusable lessons (semantic memory), injected into prompts.
- **Gates** (`urban_experiment.py`): evidence gate, rule injection gate, query-fusion gate, shot drop gate, etc.

## Design highlights (why it's built this way)

- **Non-parametric "learning"**: no fine-tuning; improvements come from better retrieval + better prompt scaffolding.
- **Two memories, two roles**:
  - Vault = instance-level few-shot (grounded examples).
  - RuleBank = general lessons (helpful when retrieval is weak).
- **Cost/stability gates**: only use expensive scaffolding when retrieval is weak (fail/low-sim), preventing "over-guidance" and negative transfer.
- **Robust structured output**: real VLM outputs are messy; parsing must be defensive.

## Install

```bash
pip install -r requirements.txt
```

Optional (faster image handling + urban filtering): `pip install pillow`.

## Quickstart (offline / mock)

Put a few images into `PIC_DATA/` and run:

```bash
python urban_experiment.py
```

## Best-practice run (urban case, folder-based)

Prereqs: configure real backends via `.env` (or env vars). See `.env.example`.

Recommended trunk config baked into the runner:

- `sim_topk(k=2) + RuleBank(k=4) + Evidence gate(fail_or_low_sim, min_sim=0.68)`
- text-only few-shot (`caption+text` query embedding, vault embeds `text`)

Run on any folder:

```bash
conda run -n Evoshot python run_best_practice_folder.py --image_dir PIC_DATA --train_n 8 --test_n 16 --seed 42 --recurse
```

Output: `bench_out/best_practice_folder_*.json` (selected files, per-image traces, summaries).

Tips:
- Use `--train_dir` / `--test_dir` to split train/test folders.
- Dataset is noisy? Keep default train filtering; clean street-only data? add `--no_filter_train`.

## Experimental findings (urban case)

Note: absolute scores drift with teacher model + sampled images; compare configs **within the same protocol**.

- **Evidence gating matters** (cross-dataset, seeds 42/43/44):
  - `EVOSHOT_EVIDENCE_GATE=fail_or_low_sim` with `EVOSHOT_EVIDENCE_MIN_SIM=0.68` -> mean `avg_teacher_score~0.7296`.
  - Files: `bench_out/evidence_gate_sweep_cross_20260115_101213.json`, `bench_out/evidence_gate_sweep_cross_20260115_103644.json`, `bench_out/evidence_gate_sweep_cross_20260115_111504.json`.
- **Image-embedding + multimodal few-shot was worse & slower**, so we keep text-only context:
  - Files: `bench_out/image_text_fewshot_ablation_20260113_160957.json` (compare A/B vs C/D).
- **RuleBank gating yields tiny, unstable gains**:
  - Best grid point `EVOSHOT_RULEBANK_MIN_SIM=0.76` gave only ~`+0.0065` overall and flips per-seed.
  - File: `bench_out/rulebank_min_sim_sweep_cross_20260117_223044.json`.
- **Query fusion/expansion didn't give stable wins**:
  - Files: `bench_out/fusion_min_sim_sweep_cross_20260116_212648.json`, `bench_out/query_fusion_v2_ablation_cross_20260116_005245.json` (and siblings).
- **Some "stronger prompting" hurt** (over-constraints / extra fields / aggressive shot gates):
  - File: `bench_out/icling_improvements_ablation_cross_20260117_135420.json`.

## Repo map (what each file is for)

- `urban_experiment.py`: core pipeline, retrieval, gating, vault/rulebank updates.
- `models/student.py`: student prompting + robust parsing.
- `models/teacher.py`: teacher judge/caption (OpenAI/OpenRouter-compatible).
- `models/filter.py`: `StudentUrbanFilter` (filters noisy datasets cheaply via downscaled images + cache).
- `models/evidence.py`: local evidence extractor (facts/uncertainties).
- `models/rules.py`: RuleBank (semantic lessons).
- `embedding_test.py`: embedding API contract + quick smoke test.
- `run_best_practice_folder.py`: one-click best-practice runner for any image folder.
- `bench_out/`: JSON outputs for experiments and sweeps.

## Minimal real-backend config

Put secrets in `.env` (git-ignored) or set environment variables:

- Teacher: `EVOSHOT_TEACHER_BACKEND=real` + `OPENROUTER_API_KEY` (recommended) or `OPENAI_API_KEY`
- Student VLM: `EVOSHOT_STUDENT_BACKEND=real`, `EVOSHOT_LLM_URL`, `EVOSHOT_LLM_MODEL`
- Embeddings: `EVOSHOT_EMBED_BACKEND=real`, `EVOSHOT_EMBED_API`

## License

MIT (see `LICENSE`).
