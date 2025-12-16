import numpy as np
from src.common.embeddings.embedder import Embedder
from src.common.retriever.load_index import load_faiss_index, load_mapping


class Retriever:
    """
    FAISS-based retriever with normalized similarity scores.
    """

    def __init__(self, base_data_dir="data", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_name)
        self.index = load_faiss_index(base_data_dir)
        self.mapping = load_mapping(base_data_dir)
        
        # Detectar tipo de índice
        self.index_type = self._detect_index_type()
        print(f"🔍 Retriever inicializado - Tipo de índice: {self.index_type}")

    def _detect_index_type(self):
        """Detecta si el índice usa L2 o Inner Product."""
        index_str = str(type(self.index))
        if "L2" in index_str or "Flat" in index_str:
            return "L2"
        elif "IP" in index_str:
            return "IP"
        else:
            return "unknown"

    def _normalize_scores(self, scores):
        """
        Normaliza scores a rango [0, 1] donde 1 = mejor match.
        Maneja overflow y casos extremos.
        """
        # Proteger contra valores inválidos
        scores = np.array(scores, dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.index_type == "L2":
            # L2: distancias, invertir y normalizar
            max_score = np.max(scores) if len(scores) > 0 else 1.0
            
            if max_score == 0 or np.isinf(max_score):
                return np.ones_like(scores)
            
            # Normalizar primero, luego invertir (más robusto)
            normalized = scores / (max_score + 1.0)  # +1.0 en vez de 1e-6
            normalized = 1.0 - np.clip(normalized, 0, 1)
            return normalized
        
        elif self.index_type == "IP":
            # Inner Product: mayor es mejor
            min_score = np.min(scores)
            max_score = np.max(scores)
            
            if max_score == min_score:
                return np.ones_like(scores) * 0.5
            
            normalized = (scores - min_score) / (max_score - min_score + 1e-6)
            return np.clip(normalized, 0, 1)
        
        else:
            # Unknown: devolver clipped a [0,1]
            return np.clip(scores, 0, 1)

    def retrieve(self, query, k=5, allowed_sections=None):
      query_vec = self.embedder.encode([query]).astype("float32")

      distances, indices = self.index.search(query_vec, k * 5)

      results = []
      fallback = []

      for dist, idx in zip(distances[0], indices[0]):
          if idx < 0 or idx >= len(self.mapping):
              continue

          frag = dict(self.mapping[idx])
          frag["score"] = float(dist)  # DISTANCIA REAL
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
