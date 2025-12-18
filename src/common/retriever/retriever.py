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
    Funciona incluso cuando aún no hay documentos indexados.
    """

    def __init__(
        self,
        base_data_dir="data",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.base_data_dir = base_data_dir
        self.embedder = Embedder(model_name)

        # 🔹 Carga segura de artefactos
        self.index = load_faiss_index(base_data_dir)
        self.mapping = load_mapping(base_data_dir)
        self.index_meta = load_index_meta(base_data_dir)

        # 🔹 Normalizar mapping → siempre lista
        if not isinstance(self.mapping, list):
            self.mapping = []

        # 🔹 Estado del índice
        if self.index is None:
            self.similarity = None
            self.index_type = "none"
            print("ℹ️ Retriever inicializado SIN índice FAISS (no hay documentos)")
        else:
            self.similarity = self.index_meta.get("similarity", "cosine")
            self.index_type = self._detect_index_type()
            print(f"🔍 Retriever listo — índice: {self.index_type}, sim: {self.similarity}")

        print(f"📂 Retriever usando base_data_dir = {self.base_data_dir}")

    def _detect_index_type(self):
        if self.index is None:
            return "none"

        name = self.index.__class__.__name__
        if "IP" in name:
            return "IP"
        if "L2" in name:
            return "L2"
        return name

    def has_index(self) -> bool:
        """Indica si el retriever está listo para buscar."""
        return self.index is not None and len(self.mapping) > 0

    def retrieve(
        self,
        query: str,
        k: int = 5,
        allowed_sections=None,
        min_score: float = 0.0
    ):
        # 🚫 No hay índice → no buscar
        if not self.has_index():
            return []

        # 🔹 Embedding del query
        query_vec = self.embedder.encode([query]).astype("float32")

        search_k = max(k * 5, k)
        scores, indices = self.index.search(query_vec, search_k)

        results = []
        fallback = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.mapping):
                continue

            frag = dict(self.mapping[idx])
            frag["score"] = float(score)

            if score < min_score:
                continue

            section = frag.get("section", "unknown")

            if allowed_sections is None or section in allowed_sections:
                results.append(frag)
            else:
                fallback.append(frag)

        # 🔹 Ordenar por score (mayor = mejor)
        results.sort(key=lambda x: x["score"], reverse=True)
        fallback.sort(key=lambda x: x["score"], reverse=True)

        # 🔹 Completar con fallback si falta
        if len(results) < k:
            results.extend(fallback[: k - len(results)])

        return results[:k]
