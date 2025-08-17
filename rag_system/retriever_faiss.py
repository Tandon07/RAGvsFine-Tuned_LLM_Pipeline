# rag_system/retriever_faiss.py
import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import PROCESSED_DATA_PATH, EMBEDDING_MODEL, FAISS_INDEX_PATH

class FAISSRetriever:
    def __init__(self, processed_path=PROCESSED_DATA_PATH, index_path=FAISS_INDEX_PATH):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.processed_path = processed_path
        self.index_path = index_path

        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"QA file not found: {processed_path}")
        with open(processed_path, "r", encoding="utf-8") as f:
            self.qa_pairs = json.load(f)

        self.questions = [q["question"] for q in self.qa_pairs]

        if os.path.exists(f"{index_path}.index"):
            self.index = faiss.read_index(f"{index_path}.index")
        else:
            self.index = self._build_index()
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, f"{index_path}.index")

    def _build_index(self):
        embeddings = self.embedder.encode(self.questions, convert_to_numpy=True, normalize_embeddings=True)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query, top_k=3):
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append({
                "question": self.qa_pairs[idx]["question"],
                "answer": self.qa_pairs[idx]["answer"],
                "score": float(score)
            })
        return results

if __name__ == "__main__":
    r = FAISSRetriever()
    print(r.retrieve("What was Globex revenue in 2024?", top_k=2))