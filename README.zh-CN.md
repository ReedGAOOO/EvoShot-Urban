# EvoShot-Urban

Evolving Few‑Shot Memory for Urban Perception Learning.

English README: `README.md`.

本项目实现了一个用于“城市感知打分”的 EvoShot 实验管线：给定街景图片（可附带一段 post 文本），让 Student（视觉语言模型）输出 `safety / vibrancy / cleanliness` 三个维度的分数与理由；再由 Teacher（更强的模型）进行严格评分与批改，并在需要时把“更好的示例”写入 Vault（few-shot 记忆库），从而实现一种不做微调的迭代式学习/纠错。

## 核心组件

- Student：`models/student.py`，支持 `mock` 或通过 OpenAI-compatible HTTP 接口调用本地/远端 VLM（`/v1/chat/completions`）。
- Teacher：`models/teacher.py`，支持 `mock` 或通过 OpenRouter / OpenAI 的 chat-completions 接口进行评分与给出改进答案（需要 API Key，**会产生费用**）。
- Vault（记忆库 + 检索）：`urban_experiment.py:MockVault`，支持 `topk` / `mmr` 检索策略；可使用 `mock` embedding 或对接本地 embedding 服务（见 `embedding/embedder.py`）。
- Captioner（可选）：`models/captioner.py`，为检索构造更强的 query（`EVOSHOT_QUERY_EMBED_MODE=caption+text`）。
- Evidence extractor（可选）：`models/evidence.py`，先抽取“可见事实”以降低幻觉，再注入 Student 提示词。
- RuleBank（可选）：`models/rules.py`，把 Teacher 给出的通用 lesson 作为“语义规则记忆”注入提示词（即使 `k=0` 也可生效）。

## 环境要求

- Python 3.9+（代码使用了 `list[str]` 等类型标注语法）

## 安装

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

可选依赖（更快的图片降采样与编码）：`pip install pillow`（未安装也能运行，会自动回退到原图 base64 编码，但可能更慢）。

## 快速开始（不需要 API / 离线）

默认后端是 `mock`（Student/Teacher/Embedding 都不联网），适合先跑通流程：

1) 在 `PIC_DATA/` 放入若干张图片（`jpg/png/webp`）。
2) 运行 demo：

```bash
python urban_experiment.py
```

也可以用一个更“脚本化”的最小例子（单张/两张图）：

```bash
python run_full_loop.py --train PIC_DATA\xxx.jpg --test PIC_DATA\yyy.jpg
```

## 使用真实后端（会产生费用）

建议用 `.env`（或系统环境变量）配置。环境变量优先生效；`.env` 的示例在 `.env.example`。

### Teacher（OpenRouter / OpenAI）

- 设置 `EVOSHOT_TEACHER_BACKEND=real`
- 推荐 OpenRouter：设置 `OPENROUTER_API_KEY`
- 或使用 OpenAI：设置 `OPENAI_API_KEY`

### Student（OpenAI-compatible VLM）

- 设置 `EVOSHOT_STUDENT_BACKEND=real`
- 配置 Student 端点与模型：
  - `EVOSHOT_LLM_URL`（默认 `http://100.76.172.42:1234/v1/chat/completions`）
  - `EVOSHOT_LLM_MODEL`（默认 `qwen3-vl-30b-a3b-thinking`）

该端点需要兼容 OpenAI chat-completions，并支持图像输入（本项目会把图片编码为 data URL，或直接传 URL/data URL）。

### Embedding 服务（可选）

若要启用真实检索 embedding（支持 text / image / image+text）：

- 设置 `EVOSHOT_EMBED_BACKEND=real`
- 设置 `EVOSHOT_EMBED_API=http://localhost:8000/embed`
- 你可以用 `python embedding_test.py` 验证服务契约（请求/返回格式见 `embedding_test.py` 注释）。

当 embedding 服务跑在 WSL / 容器中时，可能无法读取 Windows 路径 `D:\...`；可设置：

- `EVOSHOT_EMBED_IMAGE_PATH_STYLE=wsl`（会把 `D:\a\b.jpg` 转换为 `/mnt/d/a/b.jpg`）

## 常用环境变量

| 变量 | 作用 | 默认值/示例 |
|---|---|---|
| `EVOSHOT_STUDENT_BACKEND` | `mock`/`real` | `mock` |
| `EVOSHOT_TEACHER_BACKEND` | `mock`/`real`（real 会收费） | `mock` |
| `EVOSHOT_EMBED_BACKEND` | `mock`/`real` | `mock` |
| `EVOSHOT_LLM_URL` | Student VLM chat-completions URL | `http://.../v1/chat/completions` |
| `EVOSHOT_LLM_MODEL` | Student VLM 模型名 | `qwen3-vl-30b-a3b-thinking` |
| `EVOSHOT_RETRIEVAL_K` | few-shot 检索条数 | `2` |
| `EVOSHOT_RETRIEVAL_STRATEGY` | `topk`/`mmr` | `mmr` |
| `EVOSHOT_QUERY_EMBED_MODE` | query 侧 embedding 输入：`text`/`caption+text`/`image+text` 等 | `text` |
| `EVOSHOT_VAULT_EMBED_MODE` | vault 侧 embedding 输入：`text`/`image`/`image+text` | `text` |
| `EVOSHOT_STUDENT_FEWSHOT_MODE` | few-shot 注入方式：`text`/`multimodal` | `text` |
| `EVOSHOT_DISABLE_VAULT_UPDATE` | `1` 冻结学习（只评估不写入 Vault） | `0` |
| `EVOSHOT_RULEBANK_ENABLED` | 启用 RuleBank（语义规则记忆） | `0` |
| `EVOSHOT_EVIDENCE_ENABLED` | 启用 Evidence 抽取（减少幻觉） | `0` |

## 实验脚本（bench）

- `urban_experiment.py`：核心管线与 demo。
- `run_full_loop.py`：单轮/两轮最小闭环示例（带 CLI 参数）。
- `run_3_control_experiments.py`：对比不同检索策略/配置的控制实验，输出 JSON 结果。
- `run_image_text_fewshot_comparison.py`：对比 text few-shot vs multimodal few-shot（会写入 `bench_out/`）。
- `analyze_bench_results.py` / `compare_retrieval_query_modes.py`：对 bench 输出做汇总与对比分析。

部分脚本里包含本地数据集路径（例如 `H:\\...`），需要按你的机器路径修改后再运行。

## 数据与输出

默认会在 `data/` 下创建目录（可通过环境变量覆盖，见 `evoshot_env.py`）：

- `data/cache/`：caption / evidence / filter 的缓存（`.jsonl`）
- `data/logs/`：日志目录
- `data/vault/`：预留的 vault 持久化目录（当前实现主要是内存 + 日志/结果）
- `bench_out/`：各类实验脚本输出的 JSON 结果

## 本地密钥与安全

- 把密钥（如 `OPENROUTER_API_KEY` / `OPENAI_API_KEY`）放在本地 `.env`（参考 `.env.example`），或用系统环境变量注入。
- `.env` / `.env.*` 已在 `.gitignore` 中忽略；若你曾经把密钥提交到 git 历史中，请立即 **撤销并轮换** 对应 Key。

## License

MIT（见 `LICENSE`）。
