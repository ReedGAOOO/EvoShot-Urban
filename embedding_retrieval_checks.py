import os

import numpy as np

from embedding.embedder import EmbeddingTestHTTPEmbedder
from urban_experiment import MockVault, Sample, UrbanExample


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def run_sanity_check() -> None:
    embedder = EmbeddingTestHTTPEmbedder()
    v1 = np.array(embedder.encode("A busy street during the day"), dtype=np.float32)
    v1b = np.array(embedder.encode("A busy street during the day"), dtype=np.float32)
    v2 = np.array(embedder.encode("A quiet park at night with no lights"), dtype=np.float32)

    print("--- Running Embedding Sanity Check ---")
    print("Vector Dimension:", len(v1))
    print("Idempotency sim (same text twice): %.4f" % cosine_sim(v1, v1b))
    sim = cosine_sim(v1, v2)
    print("Similarity (Street vs Park): %.4f" % sim)
    if sim > 0.9:
        print("Verdict: COLLAPSE>0.9")
    elif sim < 0.6:
        print("Verdict: GOOD<0.6")
    else:
        print("Verdict: MID")


def run_mmr_check() -> None:
    vault = MockVault(use_real_embedder=True)

    examples = [
        UrbanExample(
            id="nm_1",
            image_desc="A crowded night market with neon signs and BBQ stalls.",
            post_text="Night market barbecue street food with neon lights and crowds.",
            rationale="Reference example.",
            scores={"safety": 3.0, "vibrancy": 5.0, "cleanliness": 3.0},
            tier=0,
        ),
        UrbanExample(
            id="nm_2",
            image_desc="A busy street food night market with smoke and neon.",
            post_text="Night market barbecue street food with neon lights and crowds.",
            rationale="Reference example.",
            scores={"safety": 3.2, "vibrancy": 4.8, "cleanliness": 2.8},
            tier=0,
        ),
        UrbanExample(
            id="nm_3_dirty",
            image_desc="A night market with overflowing trash bins and litter near stalls.",
            post_text="Night market barbecue area with neon lights and crowds, but trash bins overflow and litter covers the ground.",
            rationale="Reference example with boundary condition (cleanliness penalty).",
            scores={"safety": 2.8, "vibrancy": 4.8, "cleanliness": 1.8},
            tier=0,
        ),
        UrbanExample(
            id="nm_4",
            image_desc="A bustling night market street with bright signage and pedestrians.",
            post_text="Night market barbecue street food with neon lights and crowds.",
            rationale="Reference example.",
            scores={"safety": 3.0, "vibrancy": 4.9, "cleanliness": 3.1},
            tier=0,
        ),
        UrbanExample(
            id="nm_5",
            image_desc="Another angle of the same crowded night market BBQ street.",
            post_text="Night market barbecue street food with neon lights and crowds.",
            rationale="Near-duplicate reference example.",
            scores={"safety": 3.0, "vibrancy": 5.0, "cleanliness": 3.0},
            tier=0,
        ),
        UrbanExample(
            id="nm_6",
            image_desc="Close-up view of barbecue skewers in a night market crowd.",
            post_text="Night market barbecue street food with neon lights and crowds.",
            rationale="Near-duplicate reference example.",
            scores={"safety": 3.1, "vibrancy": 4.9, "cleanliness": 3.0},
            tier=0,
        ),
        UrbanExample(
            id="shop_day",
            image_desc="A clean modern shopping street during daytime with pedestrians.",
            post_text="Modern shopping street in daytime, tidy sidewalks, moderate crowds.",
            rationale="Reference example.",
            scores={"safety": 4.2, "vibrancy": 4.0, "cleanliness": 4.5},
            tier=0,
        ),
        UrbanExample(
            id="alley_night",
            image_desc="An empty dark alley at night with poor lighting and graffiti.",
            post_text="Dark back alley at night, few people, looks neglected.",
            rationale="Reference example.",
            scores={"safety": 1.5, "vibrancy": 1.5, "cleanliness": 2.0},
            tier=0,
        ),
    ]

    embed_texts = [f"[IMG] {ex.image_desc}\nText: {ex.post_text}" for ex in examples]
    vecs = vault.embedder.encode_many(embed_texts)
    for ex, vec in zip(examples, vecs):
        ex.embedding = vec
        vault.storage.append(ex)

    query = Sample(
        id="q_night_market",
        image_path="(unused)",
        post_text="Night market barbecue street food with neon lights and crowds.",
        ground_truth_sim={"safety": 3, "vibrancy": 5, "cleanliness": 3},
    )

    print("\n--- Running Retrieval Diversity Check (Top-K vs MMR) ---")
    query_text = f"Text: {query.post_text}"
    q = np.array(vault.embedder.encode(query_text), dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    mat = np.array([ex.embedding for ex in vault.storage], dtype=np.float32)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mat.dot(q)
    ranked = sorted(zip([ex.id for ex in vault.storage], sims.tolist()), key=lambda x: x[1], reverse=True)
    print("Top similarity to query:")
    for ex_id, s in ranked[:8]:
        print(" ", ex_id, "%.4f" % s)

    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "topk"
    topk = vault.retrieve(query, k=3)
    print("Top-K:", [ex.id for ex in topk])

    os.environ["EVOSHOT_RETRIEVAL_STRATEGY"] = "mmr"
    for lam in ("0.8", "0.6", "0.3"):
        os.environ["EVOSHOT_MMR_LAMBDA"] = lam
        mmr = vault.retrieve(query, k=3)
        print(f"MMR λ={lam}:", [ex.id for ex in mmr])


if __name__ == "__main__":
    run_sanity_check()
    run_mmr_check()
