"""Módulo de abstención: decide cuándo no responder."""

from src.v3_rag_advanced.config import (
    SCORE_MINIMO,
    MIN_RELEVANT_FRAGMENTS,
    MIN_QUESTION_WORDS
)

def debe_abstener(pregunta, fragmentos, score_minimo=None):
    if score_minimo is None:
        score_minimo = SCORE_MINIMO

    # 1) No se recuperó nada
    if not fragmentos:
        return True

    # 2) Ningún fragmento con score razonable
    if max(f.get("score", 0.0) for f in fragmentos) < score_minimo:
        return True

    # 3) Pregunta demasiado corta
    if len(pregunta.strip().split()) < MIN_QUESTION_WORDS:
        return True

    # 4) Verificar fragmentos realmente relevantes
    relevantes = [
        f for f in fragmentos if f.get("score", 0.0) >= score_minimo
    ]
    if len(relevantes) < MIN_RELEVANT_FRAGMENTS:
        return True

    return False
