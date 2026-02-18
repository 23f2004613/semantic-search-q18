import os
import json
import numpy as np
import faiss
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("AIPIPE_TOKEN"), base_url="https://aipipe.org/openai/v1")

app = Flask(__name__)

# Load docs once
try:
    with open("docs.json", "r") as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} docs")
except FileNotFoundError:
    print("❌ Create docs.json first!")
    docs = []
    exit()

# Embed once at startup
def get_embedding(text):
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding)

print("🔄 Embedding docs...")
doc_embeddings_list = [get_embedding(doc['content']) for doc in docs]
doc_embeddings = np.array(doc_embeddings_list, dtype=np.float32)  # FIX: Force float32
dimension = doc_embeddings.shape[1]
faiss.normalize_L2(doc_embeddings)


index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)
print("✅ Index ready! Total docs:", len(docs))

def retrieve(query, k=5):
    q_emb = get_embedding(query)
    q_vec = np.array([q_emb], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    
    # Limit k to actual docs
    actual_k = min(k, len(docs))
    scores, indices = index.search(q_vec, actual_k)
    
    results = []
    unique_docs = {}
    for j in range(actual_k):
        i = indices[0][j]
        doc_id = docs[i]["id"]
        if doc_id not in unique_docs:  # Dedupe
            unique_docs[doc_id] = True
            score = max(0.0, float(scores[0][j]))
            results.append({
                "id": doc_id,
                "content": docs[i]["content"],
                "score": score,
                "metadata": docs[i].get("metadata", {})
            })
    return results[:k]  




def rerank(candidates, query, rerank_k=3):
    scored = []
    seen_ids = set()
    for doc in candidates:
        if doc["id"] in seen_ids:
            continue
        seen_ids.add(doc["id"])
        
        prompt = f'Query: "{query}"\nDocument: "{doc["content"][:1000]}"\nRate relevance 0-10. ONLY number:'
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=3
        )
        try:
            score = float(resp.choices[0].message.content.strip()) / 10
        except:
            score = 0.0
        
        scored.append((score, doc))
    
    scored.sort(key=lambda x: x[0], reverse=True)  # ← FIXED!
    return [{"id": d["id"], "content": d["content"], "score": s, 
             "metadata": d["metadata"]} for s, d in scored[:rerank_k]]




@app.route('/search', methods=['POST'])
def search():
    start = time.time()
    data = request.json or {}
    query = data.get("query", "")
    k = data.get("k", 5)
    rerank_flag = data.get("rerank", True)
    rerank_k = data.get("rerankK", 3)
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    candidates = retrieve(query, k)
    if rerank_flag:
        candidates = rerank(candidates, query, rerank_k)
    
    latency = int((time.time() - start) * 1000)
    return jsonify({
        "results": candidates,
        "reranked": rerank_flag,
        "metrics": {"latency": latency, "totalDocs": len(docs)}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
