from src.common.embeddings.embedder import Embedder
from src.common.retriever.load_index import load_faiss_index, load_mapping


class Retriever:
    """
    FAISS-based retriever with cosine similarity.
    """

    def __init__(self, base_data_dir="data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_name)
        self.index = load_faiss_index(base_data_dir)
        self.mapping = load_mapping(base_data_dir)

    def retrieve(self, query, k=5, allowed_sections=None):
        # 1️⃣ Embedding de la query
        query_vec = self.embedder.encode([query]).astype("float32")

        # 2️⃣ Buscar más candidatos para compensar filtros
        scores, indices = self.index.search(query_vec, k * 5)

        results = []
        fallback = []

        for score, idx in zip(scores[0], indices[0]):
            frag = dict(self.mapping[idx])
            frag["score"] = float(score)
            section = frag.get("section", "unknown")

            if allowed_sections is None or section in allowed_sections:
                results.append(frag)
            else:
                fallback.append(frag)

            if len(results) == k:
                break

        if len(results) < k:
            results.extend(fallback[: k - len(results)])

        return results
