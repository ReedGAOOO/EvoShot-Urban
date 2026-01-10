import json
import os
import urllib.request

API = os.getenv("EVOSHOT_EMBED_API", "http://localhost:8000/embed")
INSTRUCTION = os.getenv("EVOSHOT_EMBED_INSTRUCTION", "Retrieve images or text relevant to the user query.")


def embed_texts(texts: list[str], *, api: str = API, instruction: str = INSTRUCTION, timeout_s: float = 120.0) -> dict:
    payload = {"inputs": [{"text": t, "instruction": instruction} for t in texts]}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(api, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


if __name__ == "__main__":
    data = embed_texts(
        [
            "A woman playing with her dog on a beach at sunset.",
            "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset...",
        ]
    )
    print("n =", data["n"], "dim =", data["dim"])
    print("first8 =", data["embeddings"][0][:8])
