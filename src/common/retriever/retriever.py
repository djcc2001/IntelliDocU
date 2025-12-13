import numpy as np
from sentence_transformers import SentenceTransformer
from .load_index import load_faiss_index, load_mapping


class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = load_faiss_index()
        self.mapping = load_mapping()

    def retrieve(self, query, k=5):
        query_vec = self.embedder.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            fragment = self.mapping[idx]
            fragment["score"] = float(score)
            results.append(fragment)

        return results
