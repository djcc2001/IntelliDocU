"""Configuración para RAG Advanced."""

# Límites de contexto
MAX_CONTEXT_CHARS = 2000  # Aumentado para mejor contexto
MAX_FRAGMENT_TEXT = 500   # Fragmentos más largos
MAX_FRAGMENTS = 10         # Máximo de fragmentos a usar

# Recuperación
TOP_K = 10                # Fragmentos a recuperar inicialmente
SCORE_MINIMO = 0.08      # Score mínimo para considerar relevante

# Abstención
MIN_RELEVANT_FRAGMENTS = 1  # Mínimo de fragmentos relevantes
MIN_QUESTION_WORDS = 2      # Palabras mínimas en pregunta válida

# Texto de abstención
# La información no está disponible en el documento.
#
ABSTENTION_TEXT = "It is not mentioned in the document."
