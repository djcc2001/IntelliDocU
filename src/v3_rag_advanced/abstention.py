"""
Modulo de abstención: decide cuándo no responder.
Implementa la logica para determinar si el sistema debe abstenerse de responder
cuando no hay suficiente evidencia o la pregunta no es valida.
"""

from src.v3_rag_advanced.config import (
    PUNTUACION_MINIMA,
    MIN_FRAGMENTOS_RELEVANTES,
    MIN_PALABRAS_PREGUNTA
)


def debe_abstener(pregunta, fragmentos, puntuacion_minima=None):
    """
    Decide si el sistema debe abstenerse de responder.
    
    Args:
        pregunta: Pregunta del usuario
        fragmentos: Lista de fragmentos recuperados
        puntuacion_minima: Puntuacion minima personalizada (opcional)
    
    Returns:
        True si debe abstenerse, False si puede intentar responder
    """
    if puntuacion_minima is None:
        puntuacion_minima = PUNTUACION_MINIMA
    
    # 1) Sin fragmentos recuperados
    if not fragmentos:
        return True
    
    # 2) TODOS los fragmentos tienen puntuacion extremadamente baja (muy estricto)
    if all(fragmento.get("score", 0.0) < 0.05 for fragmento in fragmentos):
        return True
    
    # 3) Pregunta demasiado corta o vacia
    palabras = pregunta.strip().split()
    if len(palabras) < MIN_PALABRAS_PREGUNTA:
        return True
    
    # 4) Preguntas CLARAMENTE fuera de dominio (lista reducida)
    palabras_imposibles = [
        # Solo cosas MUY obvias
        "weather", "clima hoy", "receta", "recipe", 
        "chiste", "joke", "cancion", "song",
        "hola", "hello", "hi", "hey"
    ]
    pregunta_minusculas = pregunta.lower()
    if any(palabra in pregunta_minusculas for palabra in palabras_imposibles):
        return True
    
    # 5) Al menos UN fragmento medianamente relevante
    relevantes = [f for f in fragmentos if f.get("score", 0.0) >= puntuacion_minima]
    if len(relevantes) < MIN_FRAGMENTOS_RELEVANTES:
        return True
    
    # Si pasa todos los filtros, puede intentar responder
    return False
