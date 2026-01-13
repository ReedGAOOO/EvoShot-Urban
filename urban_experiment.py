import json
import random
import logging
import time
import os
from typing import List, Dict, Optional, Literal
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from colorama import Fore, Style, init

from evoshot_env import EvoShotEnv
from models.student import HTTPStudentModel
from embedding.embedder import EmbeddingTestHTTPEmbedder
from models.teacher import OpenAITeacher, TeacherFeedback
from models.captioner import LocalVisionCaptioner

# 初始化
init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UrbanExp")

# ==========================================
# 1. 配置与定义 (configs & types)
# ==========================================


class ScoreConfig:
    """这里定义你的评分维度，避免硬编码"""

    dims = {
        "safety": {"weight": 1.0, "desc": "Perceived safety"},
        "vibrancy": {"weight": 0.8, "desc": "Activity level"},
        "cleanliness": {"weight": 0.6, "desc": "Physical maintenance"},
    }
    min_val = 1.0
    max_val = 5.0

    @staticmethod
    def compute_overall(scores: Dict[str, float]) -> float:
        """核心防坑点：Overall 必须是计算出来的，不能让模型预测"""
        total_w = sum(d["weight"] for d in ScoreConfig.dims.values())
        weighted_sum = sum(scores.get(k, 1.0) * v["weight"] for k, v in ScoreConfig.dims.items())
        return round(weighted_sum / total_w, 2)


class UrbanExample(BaseModel):
    """存储在 Vault 中的单一知识点"""

    id: str
    image_path: Optional[str] = None  # 可选：用于“图像侧”embedding / 多模态 few-shot
    image_desc: str  # 关键：只存描述，不把图片塞进 Context
    post_text: str
    rationale: str
    scores: Dict[str, float]
    # 分层管理：tier 0 = 专家/锚点, tier 1 = Teacher生成, tier 2 = 临时
    tier: Literal[0, 1, 2] = 1
    embedding: Optional[List[float]] = Field(default=None, exclude=True)  # 运行时向量
    usage_count: int = 0

    def to_prompt_str(self) -> str:
        """生成 Few-shot 文本"""
        return (
            f"Example (Tier {self.tier}):\n"
            f"Input: [IMG: {self.image_desc}] + Text: '{self.post_text}'\n"
            f"Rationale: {self.rationale}\n"
            f"Scores: {json.dumps(self.scores)}\n"
            f"---\n"
        )


class StudentOutput(BaseModel):
    """学生模型的原始输出"""

    rationale: str
    scores: Dict[str, float]
    model_confidence: float  # 模型自报的置信度


class Sample(BaseModel):
    """输入样本"""

    id: str
    image_path: str
    post_text: str
    retrieval_text: Optional[str] = None
    # 模拟真实标签（仅用于 Mock 系统中生成 Teacher 的正确答案）
    ground_truth_sim: Dict[str, float]


# ==========================================
# 2. 模拟核心组件 (Mock Implementations)
# ==========================================


class MockEmbedder:
    """模拟 CLIP/Text Embedding"""

    def encode(self, text: str) -> List[float]:
        # 实际项目中这里调用 CLIP/BERT
        # 这里返回随机向量用于演示
        return np.random.rand(128).tolist()


class MockVault:
    """模拟向量数据库 (ChromaDB/Milvus)"""

    def __init__(self, *, use_real_embedder: bool = False):
        self.storage: List[UrbanExample] = []
        self.embedder = EmbeddingTestHTTPEmbedder() if use_real_embedder else MockEmbedder()

    def add(self, example: UrbanExample):
        # 实际项目中这里会有 hash 去重
        vault_embed_mode = (os.getenv("EVOSHOT_VAULT_EMBED_MODE") or "text").strip().lower()
        if isinstance(self.embedder, EmbeddingTestHTTPEmbedder) and vault_embed_mode in {"image", "image+text"} and example.image_path:
            if vault_embed_mode == "image":
                example.embedding = self.embedder.encode_image_path(str(example.image_path))
            else:
                mix_text = f"{example.image_desc}\n{example.post_text}".strip()
                example.embedding = self.embedder.encode_image_text(str(example.image_path), mix_text)
        else:
            embed_text = f"[IMG] {example.image_desc}\nText: {example.post_text}"
            example.embedding = self.embedder.encode(embed_text)
        self.storage.append(example)
        logger.info(f"{Fore.GREEN}[Vault] Added new example: {example.id} (Tier {example.tier}){Style.RESET_ALL}")

    def cleanup_low_usage(self, min_usage: int = 5):
        """
        定期清洗接口：删除低使用频率的非专家样本
        逻辑等价于 SQL: DELETE FROM vault WHERE tier > 0 AND usage_count < min_usage
        """
        before = len(self.storage)
        self.storage = [ex for ex in self.storage if not (ex.tier > 0 and ex.usage_count < min_usage)]
        removed = before - len(self.storage)
        if removed:
            logger.info(
                f"{Fore.YELLOW}[Vault] Cleaned {removed} low-usage Tier>0 examples (min_usage={min_usage}){Style.RESET_ALL}"
            )
        else:
            logger.info(f"{Fore.YELLOW}[Vault] Cleanup skipped, no low-usage Tier>0 examples found{Style.RESET_ALL}")

    def summarize(self):
        """输出当前 Vault 状态，便于观察各 tier 的占比和使用频次"""
        if not self.storage:
            logger.info(f"{Fore.MAGENTA}[Vault] Empty vault{Style.RESET_ALL}")
            return

        logger.info(f"{Fore.MAGENTA}[Vault] Summary — total={len(self.storage)}{Style.RESET_ALL}")
        for ex in sorted(self.storage, key=lambda x: (x.tier, -x.usage_count)):
            # 将 tier/usage 与分数并排输出，便于人工检查是否需要清洗
            logger.info(
                f"  id={ex.id} | tier={ex.tier} | usage_count={ex.usage_count} | scores={ex.scores}"
            )

    def retrieve(self, query: Sample, k: int = 3) -> List[UrbanExample]:
        """
        核心防坑点：检索策略
        这里模拟：优先返回 Tier 0 (专家)，然后是 Tier 1
        """
        if k <= 0:
            return []
        if not self.storage:
            return []

        no_self_hit = (os.getenv("EVOSHOT_RETRIEVAL_NO_SELF_HIT") or "").strip().lower() in {"1", "true", "yes", "on"}
        candidates: List[UrbanExample]
        if no_self_hit and getattr(query, "id", None):
            candidates = [ex for ex in self.storage if ex.id != query.id]
        else:
            candidates = list(self.storage)

        if not candidates:
            return []

        if isinstance(self.embedder, EmbeddingTestHTTPEmbedder):
            retrieval_strategy = (os.getenv("EVOSHOT_RETRIEVAL_STRATEGY") or "mmr").strip().lower()
            mmr_lambda = float(os.getenv("EVOSHOT_MMR_LAMBDA", "0.6"))
            mmr_lambda = max(0.0, min(1.0, mmr_lambda))
            mmr_candidates = int(os.getenv("EVOSHOT_MMR_CANDIDATES", str(max(20, k * 5))))
            vault_embed_mode = (os.getenv("EVOSHOT_VAULT_EMBED_MODE") or "text").strip().lower()

            query_text_base = query.retrieval_text if getattr(query, "retrieval_text", None) else query.post_text
            query_mode = (os.getenv("EVOSHOT_QUERY_EMBED_MODE") or "text").strip().lower()
            if query_mode == "image" and getattr(query, "image_path", None):
                q = np.array(self.embedder.encode_image_path(str(query.image_path)), dtype=np.float32)
            elif query_mode in {"image+text", "image_text", "image-text"} and getattr(query, "image_path", None):
                q = np.array(self.embedder.encode_image_text(str(query.image_path), str(query_text_base)), dtype=np.float32)
            else:
                query_text = f"Text: {query_text_base}"
                q = np.array(self.embedder.encode(query_text), dtype=np.float32)

            missing_inputs: List[Dict[str, object]] = []
            missing_examples: List[UrbanExample] = []
            vecs = []
            for ex in candidates:
                if ex.embedding is None:
                    missing_examples.append(ex)
                    if vault_embed_mode in {"image", "image+text"} and ex.image_path:
                        if vault_embed_mode == "image":
                            missing_inputs.append({"image": str(ex.image_path)})
                        else:
                            mix_text = f"{ex.image_desc}\n{ex.post_text}".strip()
                            missing_inputs.append({"image": str(ex.image_path), "text": mix_text})
                    else:
                        missing_inputs.append({"text": f"[IMG] {ex.image_desc}\nText: {ex.post_text}"})
                else:
                    vecs.append(ex.embedding)

            if missing_examples:
                new_vecs = self.embedder.encode_inputs(missing_inputs)
                for ex, vec in zip(missing_examples, new_vecs):
                    ex.embedding = vec
                vecs = [ex.embedding for ex in candidates]

            mat = np.array(vecs, dtype=np.float32)
            qn = float(np.linalg.norm(q)) or 1e-9
            vn = np.linalg.norm(mat, axis=1)
            vn[vn == 0] = 1e-9
            mat_norm = mat / vn[:, None]
            q_norm = q / qn
            sims_to_query = mat_norm.dot(q_norm)

            if retrieval_strategy == "random":
                import hashlib

                global_seed = int(os.getenv("EVOSHOT_RANDOM_SEED", "0"))
                qid = str(getattr(query, "id", "") or "")
                h = hashlib.md5(qid.encode("utf-8")).hexdigest()[:8]
                seed = global_seed + int(h, 16)
                rng = random.Random(seed)
                n_pick = min(k, len(candidates))
                idxs = rng.sample(list(range(len(candidates))), k=n_pick)
                selected = [candidates[i] for i in idxs]
            elif len(candidates) <= k:
                idxs = np.argsort(sims_to_query)[::-1]
                selected = [candidates[int(i)] for i in idxs]
            elif retrieval_strategy == "topk":
                idxs = np.argsort(sims_to_query)[::-1][:k]
                selected = [candidates[int(i)] for i in idxs]
            else:
                selected_indices: List[int] = []
                ranked = np.argsort(sims_to_query)[::-1]
                candidate_indices = [int(i) for i in ranked[: min(len(ranked), mmr_candidates)]]

                for _ in range(k):
                    best_idx = -1
                    best_score = -1e9

                    for idx in candidate_indices:
                        score_rel = float(sims_to_query[idx])
                        if not selected_indices:
                            score_div = 0.0
                        else:
                            div_sims = mat_norm[selected_indices].dot(mat_norm[idx])
                            score_div = float(np.max(div_sims))

                        mmr = mmr_lambda * score_rel - (1.0 - mmr_lambda) * score_div
                        if mmr > best_score:
                            best_score = mmr
                            best_idx = idx

                    if best_idx == -1:
                        break
                    selected_indices.append(best_idx)
                    candidate_indices.remove(best_idx)

                selected = [candidates[i] for i in selected_indices]
        else:
            # 简单模拟：按 Tier 排序，然后取前 k 条
            # 实际项目中是 Vector Similarity Search
            sorted_ex = sorted(candidates, key=lambda x: x.tier)
            selected = sorted_ex[:k]

        # 记录使用频次，便于后续清洗
        for ex in selected:
            ex.usage_count += 1

        return selected


class MockStudentModel:
    """模拟本地小模型 (Ollama/LLaVA)"""

    def predict(self, sample: Sample, shots: List[UrbanExample]) -> StudentOutput:
        # 模拟：如果 shots 足够多，表现会变好 (这里用随机模拟)
        # 实际项目中：调用 HTTP API 传入 Prompt

        time.sleep(0.5)  # 模拟推理耗时

        # 模拟“学习”：如果检索到的 shots 和样本真值接近，学生就能答对
        # 这里简化为：产生一个在真值附近波动的分数
        gt = sample.ground_truth_sim
        pred_scores = {}
        for dim in ScoreConfig.dims:
            # 噪声代表学生的不确定性
            noise = random.uniform(-1.0, 1.0)
            pred_scores[dim] = round(max(1, min(5, gt[dim] + noise)), 1)

        return StudentOutput(
            rationale=f"Based on the visual cues of {sample.image_path}...",
            scores=pred_scores,
            model_confidence=random.uniform(0.6, 0.95),
        )


class MockTeacherModel:
    """模拟云端大模型 (GPT-4o)"""

    def judge(self, sample: Sample, student_out: StudentOutput) -> TeacherFeedback:
        # 模拟 Teacher 拥有“上帝视角” (读取 hidden ground truth)
        gt = sample.ground_truth_sim

        # 计算学生和真值的误差 (MSE)
        mse = sum((student_out.scores[k] - gt[k]) ** 2 for k in gt) / len(gt)
        score = max(0.0, 1.0 - (mse / 4.0))  # 简单的 score 函数

        # 决定是否入库：提高门槛以让 Teacher 更频繁介入
        should_add = score < 0.8

        return TeacherFeedback(
            score=score,
            critique="Scores deviated significantly." if mse > 1 else "Good job.",
            better_rationale="Correct rationale provided by Teacher.",
            better_scores=gt,
            should_add_to_vault=should_add,
        )

    def generate_caption(self, sample_or_image_path: object) -> str:
        image_path = getattr(sample_or_image_path, "image_path", None)
        if not image_path:
            image_path = str(sample_or_image_path)
        return f"A description of {image_path}"


# ==========================================
# 3. 核心流水线 (The Pipeline)
# ==========================================


class UrbanPipeline:
    def __init__(self):
        EvoShotEnv.load().ensure_dirs()

        use_real_embedder = (os.getenv("EVOSHOT_EMBED_BACKEND") or "mock").strip().lower() == "real"
        self.vault = MockVault(use_real_embedder=use_real_embedder)

        student_backend = (os.getenv("EVOSHOT_STUDENT_BACKEND") or "mock").strip().lower()
        if student_backend == "real":
            self.student = HTTPStudentModel(
                output_model=StudentOutput,
                score_dims=list(ScoreConfig.dims.keys()),
            )
        else:
            self.student = MockStudentModel()

        teacher_backend = (os.getenv("EVOSHOT_TEACHER_BACKEND") or "mock").strip().lower()
        if teacher_backend == "real":
            logger.info("Loading Real Teacher (OpenAI)... costs money.")
            try:
                self.teacher = OpenAITeacher()
            except ImportError as e:
                logger.error(f"Failed to load real teacher, falling back to mock: {e}")
                self.teacher = MockTeacherModel()
        else:
            self.teacher = MockTeacherModel()
        self.cfg = ScoreConfig()

        self._captioner = None
        retrieval_query_mode = (os.getenv("EVOSHOT_QUERY_EMBED_MODE") or "text").strip().lower()
        if retrieval_query_mode in {"caption", "caption+text"}:
            self._captioner = LocalVisionCaptioner()

        # --- 核心防坑：预热（冷启动）---
        # 系统启动时必须装载一些 Tier 0 的专家样本，否则一开始就是瞎猜
        self._seed_vault()

    def _seed_vault(self):
        logger.info("Seeding vault with Expert Examples...")
        seeds = [
            UrbanExample(
                id="seed_001",
                image_desc="A well-lit park at night with people walking.",
                post_text="Quiet park walk at night, well lit and populated.",
                rationale="High visibility and human presence indicate safety.",
                scores={"safety": 4.5, "vibrancy": 4.0, "cleanliness": 4.0},
                tier=0,  # 专家级
            ),
            UrbanExample(
                id="seed_002",
                image_desc="A crowded night market street with neon signs and dense foot traffic.",
                post_text="Crowded night market with neon lights and street food stalls.",
                rationale="High activity suggests vibrancy; crowds may reduce perceived safety; cleanliness varies.",
                scores={"safety": 3.0, "vibrancy": 5.0, "cleanliness": 3.0},
                tier=0,  # 专家级
            ),
            UrbanExample(
                id="seed_003",
                image_desc="An empty alley at night with poor lighting and graffiti.",
                post_text="Dark empty alley with graffiti and broken pavement.",
                rationale="Low lighting and isolation reduce safety; low activity; poor maintenance.",
                scores={"safety": 1.5, "vibrancy": 1.5, "cleanliness": 2.0},
                tier=0,  # 专家级
            ),
        ]
        for ex in seeds:
            self.vault.add(ex)

    def process_sample(self, sample: Sample) -> dict:
        logger.info(f"\n{Fore.CYAN}=== Processing Sample: {sample.id} ==={Style.RESET_ALL}")

        trace: dict = {"sample_id": sample.id}

        retrieval_query_mode = (os.getenv("EVOSHOT_QUERY_EMBED_MODE") or "text").strip().lower()
        if retrieval_query_mode in {"caption", "caption+text"}:
            if self._captioner is None:
                self._captioner = LocalVisionCaptioner()
            try:
                caption = self._captioner.caption(sample.image_path)
                if retrieval_query_mode == "caption":
                    sample.retrieval_text = caption
                else:
                    sample.retrieval_text = f"{caption}\n{sample.post_text}"
                trace["query_caption"] = caption
            except Exception as e:
                trace["query_caption_error"] = str(e)
                sample.retrieval_text = sample.post_text
        else:
            sample.retrieval_text = sample.post_text

        # 1. 检索 (Retrieval)
        retrieval_k = int(os.getenv("EVOSHOT_RETRIEVAL_K", "2"))
        retrieval_k = max(0, retrieval_k)
        shots = self.vault.retrieve(sample, k=retrieval_k)
        logger.info(f"Retrieved {len(shots)} few-shot examples: {[f'{ex.id}(t{ex.tier})' for ex in shots]}")
        trace["retrieved_ids"] = [ex.id for ex in shots]

        # 2. 学生推理 (Student Inference)
        # 实际 Prompt 组装在这里发生
        try:
            student_out = self.student.predict(sample, shots)

            # 核心防坑：计算 Overall，而不是信模型的
            overall = self.cfg.compute_overall(student_out.scores)

            logger.info(
                f"Student Pred: {student_out.scores} (Overall: {overall}) | Conf: {student_out.model_confidence:.2f}"
            )
            trace["student_scores"] = dict(student_out.scores)
            trace["student_confidence"] = float(student_out.model_confidence)

        except (ValidationError, ValueError, RuntimeError, FileNotFoundError) as e:
            logger.error(f"Student Inference/Parsing Failed: {e}")
            trace["error"] = f"student_failed: {e}"
            return trace  # 实际项目中这里要 retry

        # 3. 门控机制 (Gating / Active Learning)
        # 策略：如果置信度低，或者随机抽检，则呼叫 Teacher
        # 这里模拟：总是呼叫 Teacher 以便演示 Flow
        call_teacher = True

        if call_teacher:
            feedback = self.teacher.judge(sample, student_out)
            logger.info(f"Teacher Score: {feedback.score:.2f} | Critique: {feedback.critique}")
            trace["teacher_score"] = float(feedback.score)
            trace["teacher_should_add"] = bool(feedback.should_add_to_vault)
            trace["teacher_critique"] = str(feedback.critique)

            # 4. 动态更新 (Dynamic Update)
            if feedback.should_add_to_vault:
                disable_update = (os.getenv("EVOSHOT_DISABLE_VAULT_UPDATE") or "").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                if disable_update:
                    logger.info("Vault update disabled; skipping add.")
                    trace["vault_added_id"] = None
                    return trace
                # 核心防坑：入库的是 Teacher 的修正版，不是学生版
                caption_backend = (os.getenv("EVOSHOT_VAULT_CAPTION_BACKEND") or "teacher").strip().lower()
                caption = ""
                if caption_backend in {"student", "local", "captioner"}:
                    if self._captioner is None:
                        self._captioner = LocalVisionCaptioner()
                    try:
                        caption = self._captioner.caption(sample.image_path)
                    except Exception as e:
                        trace["vault_caption_error"] = str(e)
                        caption = ""
                else:
                    caption = self.teacher.generate_caption(sample)
                    if not caption or caption.strip().lower() == "urban scene description unavailable.":
                        if self._captioner is None:
                            self._captioner = LocalVisionCaptioner()
                        try:
                            caption = self._captioner.caption(sample.image_path)
                            trace["vault_caption_fallback"] = "student"
                        except Exception as e:
                            trace["vault_caption_error"] = str(e)
                            caption = caption or "Urban scene description unavailable."
                new_ex = UrbanExample(
                    id=sample.id,
                    image_path=sample.image_path,
                    image_desc=caption,
                    post_text=sample.post_text,
                    rationale=feedback.better_rationale,
                    scores=feedback.better_scores,
                    tier=1,  # Teacher 生成级
                )
                self.vault.add(new_ex)
                trace["vault_added_id"] = new_ex.id
            else:
                logger.info("Sample skipped (Student did well enough or Teacher unsure).")
                trace["vault_added_id"] = None

        return trace


# ==========================================
# 4. 运行实验
# ==========================================


if __name__ == "__main__":
    pipeline = UrbanPipeline()

    # 模拟数据流：随着数据进来，Vault 应该会变大，理论上 Student 也会变准
    # 这里生成 5 个测试样本
    pic_dir = Path(__file__).resolve().parent / "PIC_DATA"
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = sorted([p for p in pic_dir.glob("*") if p.is_file() and p.suffix.lower() in exts])
    if not image_files:
        raise FileNotFoundError(f"No image files found in {pic_dir}. Expected one of: {sorted(exts)}")

    test_samples = [
        Sample(
            id="test_1",
            image_path=str(image_files[0 % len(image_files)]),
            post_text="Quiet park walk at night under bright streetlights.",
            ground_truth_sim={"safety": random.randint(1, 5), "vibrancy": 3, "cleanliness": 4},
        ),
        Sample(
            id="test_2",
            image_path=str(image_files[1 % len(image_files)]),
            post_text="Crowded night market with neon lights and street food stalls.",
            ground_truth_sim={"safety": random.randint(1, 5), "vibrancy": 3, "cleanliness": 4},
        ),
        Sample(
            id="test_3",
            image_path=str(image_files[2 % len(image_files)]),
            post_text="Dark empty alley with graffiti and broken pavement.",
            ground_truth_sim={"safety": random.randint(1, 5), "vibrancy": 3, "cleanliness": 4},
        ),
        Sample(
            id="test_4",
            image_path=str(image_files[3 % len(image_files)]),
            post_text="Clean modern shopping street during the day with pedestrians.",
            ground_truth_sim={"safety": random.randint(1, 5), "vibrancy": 3, "cleanliness": 4},
        ),
        Sample(
            id="test_5",
            image_path=str(image_files[4 % len(image_files)]),
            post_text="Residential street with moderate foot traffic and tidy sidewalks.",
            ground_truth_sim={"safety": random.randint(1, 5), "vibrancy": 3, "cleanliness": 4},
        ),
    ]

    for s in test_samples:
        pipeline.process_sample(s)
        time.sleep(0.5)

    # 演示：定期清洗低频使用的 Tier>0 样本，保持专家锚点纯净
    pipeline.vault.cleanup_low_usage(min_usage=5)

    # 输出清洗后的 Vault 状态，确保 Tier 结构可观察
    pipeline.vault.summarize()

    logger.info(f"\nFinal Vault Size: {len(pipeline.vault.storage)}")
