"""
Configuracion para RAG Avanzado (Version 3).
Define limites y parametros para recuperacion, contexto y abstención.
"""

# Limites de contexto
MAX_CARACTERES_CONTEXTO = 2000  # Aumentado para mejor contexto
MAX_TEXTO_FRAGMENTO = 500   # Fragmentos mas largos
MAX_FRAGMENTOS = 5         # Maximo de fragmentos a usar

# Recuperacion
TOP_K = 10                # Fragmentos a recuperar inicialmente
PUNTUACION_MINIMA = 0.05       # Puntuacion minima para considerar relevante

# Abstención
MIN_FRAGMENTOS_RELEVANTES = 1  # Minimo de fragmentos relevantes (reducido)
MIN_PALABRAS_PREGUNTA = 2      # Palabras minimas en pregunta valida (reducido)

# Texto de abstención
TEXTO_ABSTENCION = "No se menciona en el documento."


# Alias para mantener compatibilidad con codigo existente
MAX_CONTEXT_CHARS = MAX_CARACTERES_CONTEXTO
MAX_FRAGMENT_TEXT = MAX_TEXTO_FRAGMENTO
SCORE_MINIMO = PUNTUACION_MINIMA
MIN_RELEVANT_FRAGMENTS = MIN_FRAGMENTOS_RELEVANTES
MIN_QUESTION_WORDS = MIN_PALABRAS_PREGUNTA
ABSTENTION_TEXT = TEXTO_ABSTENCION
