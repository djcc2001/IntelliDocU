import numpy as np
from sentence_transformers import SentenceTransformer
from .load_index import load_faiss_index, load_mapping


class Retriever:
    """
    Retriever basado en FAISS + SentenceTransformers (cosine similarity).

    Requisitos:
    - Índice FAISS: IndexFlatIP
    - Embeddings normalizados
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = load_faiss_index()
        self.mapping = load_mapping()

    def _encode_and_normalize(self, texts, batch_size=16):
        """
        Convierte una lista de textos en embeddings normalizados
        """
        vectors = self.embedder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vectors.astype("float32")

    def retrieve(self, query, k=5, allowed_sections=None):
        """
        Recupera los k fragments más relevantes para la query.
        Si allowed_sections está definido, filtra primero por sección.
        """
        # 1. Generar embedding de la query
        query_vec = self._encode_and_normalize([query])

        # 2. Buscar más candidatos para compensar filtrado por sección
        scores, indices = self.index.search(query_vec, k * 5)

        results = []
        fallback = []

        for score, idx in zip(scores[0], indices[0]):
            fragment = dict(self.mapping[idx])
            fragment["score"] = float(score)
            section = fragment.get("section", "unknown")

            if allowed_sections is None or section in allowed_sections:
                results.append(fragment)
            else:
                fallback.append(fragment)

            if len(results) == k:
                break

        # Si no hay suficientes, rellenar con fallback
        if len(results) < k:
            results.extend(fallback[: k - len(results)])

        return results
