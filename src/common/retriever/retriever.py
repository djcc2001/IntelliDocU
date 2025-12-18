import numpy as np
from src.common.embeddings.embedder import Embedder
from src.common.retriever.load_index import (
    load_faiss_index,
    load_mapping,
    load_index_meta
)


class Retriever:
    """
    FAISS-based retriever (cosine similarity).
    Compatible con el pipeline completo de chunking + embeddings.
    """

    def __init__(
        self,
        base_data_dir="data",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.embedder = Embedder(model_name)
        self.index = load_faiss_index(base_data_dir)
        self.mapping = load_mapping(base_data_dir)
        self.index_meta = load_index_meta(base_data_dir)

        self.similarity = self.index_meta.get("similarity", "cosine")

        # Detectar tipo REAL de índice
        self.index_type = self._detect_index_type()
        print(f"🔍 Retriever listo — índice: {self.index_type}, sim: {self.similarity}")

    def _detect_index_type(self):
        if isinstance(self.index, type(self.index)):
            # FAISS no expone bien el tipo, usamos class name
            name = self.index.__class__.__name__
            if "IP" in name:
                return "IP"
            if "L2" in name:
                return "L2"
        return "unknown"

    def retrieve(
        self,
        query: str,
        k: int = 5,
        allowed_sections=None,
        min_score: float = 0.25
    ):
        """
        Recupera los k chunks más relevantes.

        Args:
            query: pregunta del usuario
            k: número de chunks finales
            allowed_sections: lista de secciones permitidas
            min_score: umbral mínimo de similitud (cosine)

        Returns:
            Lista de dicts con metadata + score
        """
        query_vec = self.embedder.encode([query]).astype("float32")

        # Buscar más para poder filtrar
        search_k = max(k * 5, k)
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        fallback = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.mapping):
                continue

            frag = dict(self.mapping[idx])
            frag["score"] = float(score)  # cosine similarity

            # Filtrar por score mínimo
            if score < min_score:
                continue

            section = frag.get("section", "unknown")

            if allowed_sections is None or section in allowed_sections:
                results.append(frag)
            else:
                fallback.append(frag)

        # Ordenar explícitamente por score descendente
        results.sort(key=lambda x: x["score"], reverse=True)
        fallback.sort(key=lambda x: x["score"], reverse=True)

        # Completar con fallback si falta
        if len(results) < k:
            results.extend(fallback[: k - len(results)])

        return results[:k]
