# EvoShot-Urban

EvoShot（Evolutionary Few‑Shot System）：Teacher–Student + 检索增强的 in-context 学习框架（**不做 fine-tune**）。

本仓库提供首个案例应用：**城市环境感知**（展示“低成本云端 Teacher 稀疏指导本地中小 Student”的降本增效思路；成本估算见下文）。

English README：`README.md`。

## EvoShot 框架（一段话讲清楚）

EvoShot 是一种“非参数学习闭环”：本地/便宜的 **Student** 负责大多数推理；更强的（云端）**Teacher** 只在必要时介入判卷与纠错；系统把纠错结果写入可检索的 few-shot 记忆（Vault + RuleBank），下次遇到相似输入时通过检索注入 prompt 来提升质量，从而减少 Teacher 调用次数与总成本。框架本身是通用的：只需替换输出 schema、Teacher rubric（判卷标准）与记忆格式，即可迁移到其它任务。

## 一眼看懂（城市案例）

效果（Teacher reward 记为 `[0,1]`，越高越好）：

- 纯 Student（无检索 / 无 RuleBank / 无 Evidence）：`avg_teacher_score=0.54`，`pass_rate>=0.8=0%`
- EvoShot 最佳主干：`avg_teacher_score=0.762`（**+0.222**，约 `+41%`；约等于“满分”的 `76%`），`pass_rate>=0.8=30%`

来源（同一协议、同一批次）：`bench_out/rulebank_evidence_ablation_cross_20260114_201832.json`。

稳定性补充：在 cross-dataset 的 3 批次 sweep 中，采用当前主干阈值（`EVOSHOT_EVIDENCE_GATE=fail_or_low_sim`、`EVOSHOT_EVIDENCE_MIN_SIM=0.68`）可达到均值 `avg_teacher_score≈0.7296`（来源见下）。

成本（粗略估算，把“每张图一次云端 Teacher 调用”的成本记为 `x`）：

- 纯云端 Teacher（`N` 张图）：成本 ≈ `N * x`
- EvoShot：所有 `N` 张图都走本地 Student；Teacher 只对一部分输入 `p` 介入（低相似度/低置信度/抽检），成本 ≈ `p * N * x`
  - 在 cross-dataset 协议里，`EVOSHOT_EVIDENCE_MIN_SIM=0.68` 下“检索偏弱→触发 scaffolding”的比例约 `0.40`，如果按同一信号门控 Teacher，则可减少约 `60%` 的 Teacher 调用（评测脚本为了打分仍会调用 Teacher）。
  - 例子：`N=10,000` 张图 → 纯 Teacher ≈ `10,000*x`；EvoShot 若 `p=0.40` ≈ `4,000*x`（约 `2.5×` 更省）。
  - 来源：`bench_out/evidence_gate_sweep_cross_20260115_101213.json`、`bench_out/evidence_gate_sweep_cross_20260115_103644.json`、`bench_out/evidence_gate_sweep_cross_20260115_111504.json`。
  - 实用提示：评测脚本为了算 `avg_teacher_score` 会调用 Teacher；真实低成本部署时可主要运行 Student+检索/记忆，把 Teacher 留给主动学习或抽检。

## 实验中使用的模型

目前 `bench_out/*` 中引用的实验结果所用模型为：

- Teacher（云端）：`x-ai/grok-4.1-fast`（也可简称 `grok4.1fast`，OpenRouter），`EVOSHOT_TEACHER_TEMPERATURE=0`
- Student（本地 LM Studio）：`unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF-Q8_k_xl`，`EVOSHOT_LLM_TEMPERATURE=0`

模型均可通过 `.env` / 环境变量替换。

### 本仓库的首个案例：城市环境感知打分

输入图片（可附带 post 文本），Student 输出结构化分数：

- `safety / vibrancy / cleanliness`（1.0–5.0）

## 架构与亮点（Teacher–Student + Memory）

- **Student**：`models/student.py`（OpenAI-compatible `/v1/chat/completions` + `robust_json_parser()` 防脏 JSON）。
- **Teacher**：`models/teacher.py`（严厉打分 + 改进 rationale/scores + 可选 caption；真实 Teacher **会产生费用**）。
- **Vault**：`urban_experiment.py`（Tier-0 seeds + Tier-1 teacher-corrected；`topk/mmr` 检索）。
- **Embedding**：按 `embedding_test.py` 的契约对接 embedding 服务（当前最佳实践主要用 text embedding 做检索）。
- **Evidence**：`models/evidence.py` 抽取“可见事实”降低幻觉。
- **RuleBank**：`models/rules.py` 存储短 lesson，在检索弱时提供通用提示。
- **门控机制**：在 `urban_experiment.py` 内对 evidence / rule 注入 / query fusion / shots drop 等做门控，控制成本并减少负迁移。

### 最佳主干结构图（含门控）

```mermaid
flowchart TD
  A[输入: 图片 + post_text] --> B[Captioner (本地)]
  B --> C[Query embedding (caption+text)]
  C --> D[Vault 检索 shots (top-k=2)]
  D --> E{Shot gate: max_sim < EVOSHOT_SHOTS_MIN_SIM?}
  E -- 是 --> F[丢弃 shots (k=0)]
  E -- 否 --> G[保留 shots]
  F --> H{Evidence gate: 检索失败 OR max_sim < EVOSHOT_EVIDENCE_MIN_SIM?}
  G --> H
  H -- 是 --> I[Evidence 抽取器 (本地)]
  H -- 否 --> J[跳过 evidence]
  I --> K{RuleBank gate: 注入 rules?}
  J --> K
  K -- 是 --> L[选择 rules (k=4)]
  K -- 否 --> M[不注入 rules]
  L --> N[组装 prompt: shots + rules + evidence]
  M --> N
  N --> O[Student VLM (本地, JSON)]
  O --> P[Teacher 判卷 (云端)]
  P --> Q{should_add_to_vault?}
  Q -- 是 --> R[写入 Vault(Tier1) + 写入 RuleBank(lesson)]
  Q -- 否 --> S[不更新]
```

主要门控（均在 `urban_experiment.py`）：

- Evidence gate：`EVOSHOT_EVIDENCE_ENABLED`、`EVOSHOT_EVIDENCE_GATE`、`EVOSHOT_EVIDENCE_MIN_SIM`
- Rule 注入 gate：`EVOSHOT_RULEBANK_ENABLED`、`EVOSHOT_RULEBANK_GATE`、`EVOSHOT_RULEBANK_MIN_SIM`
- Shot gate（防负迁移）：`EVOSHOT_SHOTS_MIN_SIM`
- Query expansion / fusion（可选）：`EVOSHOT_RETRIEVAL_QUERY_EXPANSION`、`EVOSHOT_RETRIEVAL_QUERY_FUSION`、`EVOSHOT_RETRIEVAL_QUERY_FUSION_*`
- Self-revision（二阶段可选）：`EVOSHOT_SELF_REVISION_ENABLED`、`EVOSHOT_SELF_REVISION_*`
- 记忆更新门控：`EVOSHOT_DISABLE_VAULT_UPDATE`、`EVOSHOT_VAULT_REQUIRE_LESSON`

## 安装

```bash
pip install -r requirements.txt
```

可选（更快的图片处理 + urban filter）：`pip install pillow`。

## 快速开始（离线 / mock）

把几张图片放到 `PIC_DATA/`，然后运行：

```bash
python urban_experiment.py
```

## 当前最佳实践（城市案例，可一键跑）

推荐主干配置（脚本内已固化）：

- `sim_topk(k=2) + RuleBank(k=4) + Evidence gate(fail_or_low_sim, min_sim=0.68)`
- text-only few-shot（`caption+text` 做 query embedding，Vault embed `text`）

对任意图片文件夹一键运行：

```bash
conda run -n Evoshot python run_best_practice_folder.py --image_dir PIC_DATA --train_n 8 --test_n 16 --seed 42 --recurse
```

输出：`bench_out/best_practice_folder_*.json`（包含采样到的图片、每张图的 trace、汇总指标）。

小贴士：
- 需要分开训练/测试目录：用 `--train_dir` / `--test_dir`。
- 训练集很干净（全是街景）：可加 `--no_filter_train` 跳过过滤加速。

最小 `.env`（不要提交密钥到仓库）：

```env
# Teacher（云端）
EVOSHOT_TEACHER_BACKEND=real
EVOSHOT_TEACHER_API_URL=https://openrouter.ai/api/v1/chat/completions
EVOSHOT_TEACHER_MODEL=x-ai/grok-4.1-fast
OPENROUTER_API_KEY=sk-...

# Student（本地，LM Studio 的 OpenAI-compatible server）
EVOSHOT_STUDENT_BACKEND=real
EVOSHOT_LLM_URL=http://localhost:1234/v1/chat/completions
EVOSHOT_LLM_MODEL=unsloth/Qwen3-VL-30B-A3B-Thinking-GGUF-Q8_k_xl
EVOSHOT_LLM_SEND_IMAGE=1

# Embedding（需满足 embedding_test.py 契约）
EVOSHOT_EMBED_BACKEND=real
EVOSHOT_EMBED_API=http://localhost:8000/embed
```

输入怎么配：

- 图片目录：`run_best_practice_folder.py --image_dir ...`（或 `--train_dir/--test_dir`）
- post 文本：`run_best_practice_folder.py --post_text "..."`（或设置 `EVOSHOT_DEFAULT_POST_TEXT`）

LM Studio 快速自检：

- 服务是否启动：`curl http://localhost:1234/v1/models`
- 视觉输入：本项目会把图片编码为 `image_url` 的 data URL（见 `models/student.py`）

Embedding 快速自检：

- `python embedding_test.py`（要求 `EVOSHOT_EMBED_API` 可访问）

## 实验结论（城市案例，用实验解释“为什么这样设计”）

注意：Teacher 模型/图片采样变化会导致绝对分数漂移；请在同一协议内看相对差异。

- **Evidence 的门控是关键收益来源**（跨数据集 + 多 seed）  
  `EVOSHOT_EVIDENCE_GATE=fail_or_low_sim` 且 `EVOSHOT_EVIDENCE_MIN_SIM=0.68` 在 seeds(42/43/44) 上 `avg_teacher_score` 均值约 **0.7296**。  
  文件：`bench_out/evidence_gate_sweep_cross_20260115_101213.json`、`bench_out/evidence_gate_sweep_cross_20260115_103644.json`、`bench_out/evidence_gate_sweep_cross_20260115_111504.json`。
- **图像 embedding + multimodal few-shot 更慢且更差**，因此最佳实践回到原始 text context 形态：  
  文件：`bench_out/image_text_fewshot_ablation_20260113_160957.json`（A/B 明显优于 C/D）。
- **RuleBank 做门控收益很小且不稳定**（不同 seed 可能正负翻转），所以默认常开更稳：  
  文件：`bench_out/rulebank_min_sim_sweep_cross_20260117_223044.json`。
- **query fusion / query expansion 没有稳定正收益**，默认关闭：  
  文件：`bench_out/fusion_min_sim_sweep_cross_20260116_212648.json`、`bench_out/query_fusion_v2_ablation_cross_20260116_005245.json`（及同系列）。
- **过强的 prompt 约束/额外字段/激进 shot gate 可能伤害性能**：  
  文件：`bench_out/icling_improvements_ablation_cross_20260117_135420.json`。

## 代码地图（每个文件干什么）

- `urban_experiment.py`：核心流水线、检索、门控、Vault/RuleBank 动态更新。
- `models/student.py`：Student 调用与结构化输出解析。
- `models/teacher.py`：Teacher 判卷/生成改进答案与 caption。
- `models/filter.py`：`StudentUrbanFilter`（大数据集先过滤噪声，含降采样与缓存）。
- `models/evidence.py`：evidence 抽取器（facts/uncertainties）。
- `models/rules.py`：RuleBank（lesson 语义记忆）。
- `embedding_test.py`：embedding 服务契约与快速自检。
- `run_best_practice_folder.py`：一键跑最佳实践（可指定任意图片文件夹）。
- `bench_out/`：实验输出 JSON（各种 sweep/ablation 的结果）。

## 最小真实后端配置

把密钥写在本地 `.env`（已 gitignore，见 `.env.example`）或系统环境变量：

- Teacher：`EVOSHOT_TEACHER_BACKEND=real` + `OPENROUTER_API_KEY`（推荐）或 `OPENAI_API_KEY`
- Student VLM：`EVOSHOT_STUDENT_BACKEND=real`，以及 `EVOSHOT_LLM_URL`、`EVOSHOT_LLM_MODEL`
- Embedding：`EVOSHOT_EMBED_BACKEND=real`，以及 `EVOSHOT_EMBED_API`

## License

MIT（见 `LICENSE`）。
