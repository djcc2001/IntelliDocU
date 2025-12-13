# src/v3_rag_advanced/abstention.py

def debe_abstener(pregunta, fragmentos, score_minimo=0.25):
    """
    Decide si el sistema debe abstenerse de responder
    """

    if not fragmentos:
        return True

    mejor_score = fragmentos[0]["score"]

    if mejor_score < score_minimo:
        return True

    # Heuristica simple para preguntas imposibles
    palabras_imposibles = ["quantum", "weather", "capital of", "president"]
    pregunta_lower = pregunta.lower()

    for palabra in palabras_imposibles:
        if palabra in pregunta_lower:
            return True

    return False
