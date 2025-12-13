from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        # Carga del modelo de embeddings
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size=16):
        """
        Convierte una lista de textos en embeddings normalizados
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
