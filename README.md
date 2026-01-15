# EvoShot-Urban

Evolving Few‑Shot Memory for Urban Perception Learning.

Chinese README: `README.zh-CN.md`.

## Overview

This project implements an EvoShot-style experimental pipeline for **urban perception scoring**. Given an image (optionally with a short “post text”), the Student (a VLM) predicts:

- `safety`
- `vibrancy`
- `cleanliness`

Then a Teacher model grades the Student output strictly and, when needed, writes an improved example into a Vault (few-shot memory). This enables iterative correction/learning **without fine-tuning**.

## Key components

- Student: `models/student.py` (`mock` or OpenAI-compatible HTTP VLM via `/v1/chat/completions`)
- Teacher: `models/teacher.py` (`mock` or OpenRouter/OpenAI chat-completions; **costs money** when real)
- Vault + retrieval: `urban_experiment.py` (`topk` / `mmr`; mock embeddings or a local embedding service via `embedding/embedder.py`)
- Captioner (optional): `models/captioner.py` for stronger retrieval query text (`EVOSHOT_QUERY_EMBED_MODE=caption+text`)
- Evidence extraction (optional): `models/evidence.py` for grounded “visible facts” to reduce hallucinations
- RuleBank (optional): `models/rules.py` stores reusable lessons from Teacher feedback and injects them into prompts

## Requirements

- Python 3.9+

## Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional (faster image downscaling/encoding): `pip install pillow`.

## Quickstart (offline / no API)

By default everything runs in `mock` mode (no network), which is useful to validate the pipeline end-to-end.

1) Put a few images into `PIC_DATA/` (`jpg/png/webp`).
2) Run the demo:

```bash
python urban_experiment.py
```

Or run a minimal loop with explicit images:

```bash
python run_full_loop.py --train PIC_DATA\xxx.jpg --test PIC_DATA\yyy.jpg
```

## Real backends (costs money)

Use `.env` (or system env vars). Env vars take precedence; see `.env.example`.

### Teacher (OpenRouter / OpenAI)

- Set `EVOSHOT_TEACHER_BACKEND=real`
- Prefer OpenRouter: set `OPENROUTER_API_KEY`
- Or OpenAI: set `OPENAI_API_KEY`

### Student (OpenAI-compatible VLM)

- Set `EVOSHOT_STUDENT_BACKEND=real`
- Configure:
  - `EVOSHOT_LLM_URL` (default `http://100.76.172.42:1234/v1/chat/completions`)
  - `EVOSHOT_LLM_MODEL` (default `qwen3-vl-30b-a3b-thinking`)

The endpoint must support OpenAI chat-completions and image inputs (this project sends images as data URLs, or passes through image URLs/data URLs).

### Embedding service (optional)

To enable real embeddings for retrieval (text / image / image+text):

- Set `EVOSHOT_EMBED_BACKEND=real`
- Set `EVOSHOT_EMBED_API=http://localhost:8000/embed`
- Validate the contract with `python embedding_test.py` (request/response format is in `embedding_test.py`).

If the embedding service runs in WSL/container and can’t read `D:\...` paths:

- Set `EVOSHOT_EMBED_IMAGE_PATH_STYLE=wsl` (converts to `/mnt/<drive>/...`)

## Common environment variables

| Variable | Meaning | Default / Example |
|---|---|---|
| `EVOSHOT_STUDENT_BACKEND` | `mock` / `real` | `mock` |
| `EVOSHOT_TEACHER_BACKEND` | `mock` / `real` (real costs money) | `mock` |
| `EVOSHOT_EMBED_BACKEND` | `mock` / `real` | `mock` |
| `EVOSHOT_LLM_URL` | Student chat-completions URL | `http://.../v1/chat/completions` |
| `EVOSHOT_LLM_MODEL` | Student model name | `qwen3-vl-30b-a3b-thinking` |
| `EVOSHOT_RETRIEVAL_K` | few-shot examples to retrieve | `2` |
| `EVOSHOT_RETRIEVAL_STRATEGY` | `topk` / `mmr` | `mmr` |
| `EVOSHOT_QUERY_EMBED_MODE` | query embedding mode: `text` / `caption+text` / `image+text` | `text` |
| `EVOSHOT_VAULT_EMBED_MODE` | vault embedding mode: `text` / `image` / `image+text` | `text` |
| `EVOSHOT_STUDENT_FEWSHOT_MODE` | few-shot injection: `text` / `multimodal` | `text` |
| `EVOSHOT_DISABLE_VAULT_UPDATE` | freeze learning when `1` | `0` |
| `EVOSHOT_RULEBANK_ENABLED` | enable RuleBank | `0` |
| `EVOSHOT_EVIDENCE_ENABLED` | enable evidence extraction | `0` |

## Experiment scripts

- `urban_experiment.py`: main pipeline + demo
- `run_full_loop.py`: minimal CLI loop
- `run_3_control_experiments.py`: controlled experiments across configs, outputs JSON
- `run_image_text_fewshot_comparison.py`: compares text few-shot vs multimodal few-shot (writes to `bench_out/`)
- `analyze_bench_results.py` / `compare_retrieval_query_modes.py`: summarize/compare bench outputs

Some scripts contain machine-specific dataset paths (e.g. `H:\\...`) and need local edits before running.

## Data & outputs

Default directories under `data/` (overridable; see `evoshot_env.py`):

- `data/cache/`: caption/evidence/filter caches (`.jsonl`)
- `data/logs/`: logs
- `data/vault/`: reserved for vault persistence (current implementation is mostly in-memory + outputs)
- `bench_out/`: experiment JSON outputs

## Local secrets

- Put secrets (e.g. `OPENROUTER_API_KEY` / `OPENAI_API_KEY`) in a local `.env` (see `.env.example`) or as environment variables.
- `.env` / `.env.*` are git-ignored; if a key was ever committed to git history, revoke & rotate it immediately.

## License

MIT (see `LICENSE`).
