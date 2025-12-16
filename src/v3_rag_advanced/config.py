#config.py:
"""Configuración para RAG Advanced."""

# Límites de contexto
MAX_CONTEXT_CHARS = 2000  # Aumentado para mejor contexto
MAX_FRAGMENT_TEXT = 500   # Fragmentos más largos
MAX_FRAGMENTS = 5         # Máximo de fragmentos a usar

# Recuperación
TOP_K = 10                # Fragmentos a recuperar inicialmente
SCORE_MINIMO = 0.05       # Score mínimo para considerar relevante

# Abstención
MIN_RELEVANT_FRAGMENTS = 1  # Mínimo de fragmentos relevantes (reducido)
MIN_QUESTION_WORDS = 2      # Palabras mínimas en pregunta válida (reducido)

# Texto de abstención
ABSTENTION_TEXT = "It is not mentioned in the document."
