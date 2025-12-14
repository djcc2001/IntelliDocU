def debe_abstener(pregunta, fragmentos, score_minimo=0.15):
    """
    Decide si la pregunta no tiene respuesta en los fragmentos disponibles.
    """

    # 1️⃣ No hay fragmentos
    if not fragmentos:
        return True

    # 2️⃣ Ningún fragmento relevante según score
    if all(f["score"] < score_minimo for f in fragmentos):
        return True

    # 3️⃣ Preguntas imposibles o fuera de dominio
    palabras_imposibles = [
        "quantum", "weather", "capital of", "president", "hola", "como estas", "chiste"
    ]
    pregunta_lower = pregunta.lower()
    if any(p in pregunta_lower for p in palabras_imposibles):
        return True

    # 4️⃣ Opcional: abstenerse si pregunta demasiado corta o sin keywords
    if len(pregunta.split()) < 3:
        return True

    # 5️⃣ Si fragmentos relevantes son muy pocos (<2)
    fragmentos_relevantes = [f for f in fragmentos if f["score"] >= score_minimo]
    if len(fragmentos_relevantes) < 2:
        return True

    # Si pasó todos los filtros, no se abstiene
    return False
