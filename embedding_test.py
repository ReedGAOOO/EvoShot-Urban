import requests

API = "http://localhost:8000/embed"
payload = {
    "inputs": [
        {"text": "A woman playing with her dog on a beach at sunset.",
         "instruction": "Retrieve images or text relevant to the user query."},
        {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset..."}
    ]
}

r = requests.post(API, json=payload, timeout=120)
r.raise_for_status()
data = r.json()

print("n =", data["n"], "dim =", data["dim"])
print("first8 =", data["embeddings"][0][:8])
