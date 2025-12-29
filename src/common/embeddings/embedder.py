"""
Modulo para generar embeddings de texto usando modelos de sentence transformers.
Los embeddings son representaciones vectoriales del texto que permiten busqueda semantica.
"""

from sentence_transformers import SentenceTransformer


class GeneradorEmbeddings:
    """
    Clase para generar embeddings de texto usando modelos pre-entrenados.
    """
    
    def __init__(self, nombre_modelo="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings.
        
        Args:
            nombre_modelo: Nombre del modelo de sentence transformers a utilizar
        """
        self.nombre_modelo = nombre_modelo
        self.modelo = SentenceTransformer(nombre_modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()

    def codificar(self, textos, tamano_lote=32):
        """
        Genera embeddings para una lista de textos.
        
        Args:
            textos: Lista de textos a codificar
            tamano_lote: Tamano del lote para procesamiento (mayor = mas rapido pero mas memoria)
        
        Returns:
            Array numpy con los embeddings normalizados
        """
        return self.modelo.encode(
            textos,
            batch_size=tamano_lote,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalizar para usar similitud coseno
        )


# Alias para mantener compatibilidad con codigo existente
Embedder = GeneradorEmbeddings
