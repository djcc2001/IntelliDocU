#abstention.py:
"""Módulo de abstención: decide cuándo no responder."""

from src.v3_rag_advanced.config import (
    SCORE_MINIMO,
    MIN_RELEVANT_FRAGMENTS,
    MIN_QUESTION_WORDS
)

def debe_abstener(pregunta, fragmentos, score_minimo=None):
    """
    Decide si el sistema debe abstenerse de responder.
    
    Args:
        pregunta: Pregunta del usuario
        fragmentos: Lista de fragmentos recuperados
        score_minimo: Score mínimo personalizado (opcional)
    
    Returns:
        True si debe abstenerse, False si puede intentar responder
    """
    if score_minimo is None:
        score_minimo = SCORE_MINIMO
    
    # 1) Sin fragmentos recuperados
    if not fragmentos:
        return True
    
    # 2) TODOS los fragmentos tienen score extremadamente bajo (muy estricto)
    if all(f.get("score", 0.0) < 0.05 for f in fragmentos):
        return True
    
    # 3) Pregunta demasiado corta o vacía
    palabras = pregunta.strip().split()
    if len(palabras) < MIN_QUESTION_WORDS:
        return True
    
    # 4) Preguntas CLARAMENTE fuera de dominio (lista reducida)
    palabras_imposibles = [
        # Solo cosas MUY obvias
        "weather", "clima hoy", "receta", "recipe", 
        "chiste", "joke", "canción", "song",
        "hola", "hello", "hi", "hey"
    ]
    pregunta_lower = pregunta.lower()
    if any(palabra in pregunta_lower for palabra in palabras_imposibles):
        return True
    
    # 5) Al menos UN fragmento medianamente relevante
    relevantes = [f for f in fragmentos if f.get("score", 0.0) >= score_minimo]
    if len(relevantes) < MIN_RELEVANT_FRAGMENTS:
        return True
    
    # Si pasa todos los filtros, puede intentar responder
    return False
