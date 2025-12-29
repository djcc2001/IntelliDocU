"""
Modulo de construccion de prompts para RAG Basico.
Define las plantillas y limites para construir el contexto y prompt final.
"""

# Limites de contexto
MAX_CARACTERES_CONTEXTO = 1200
MAX_FRAGMENTOS = 5


def construir_contexto_literal(fragmentos):
    """
    Construye un contexto literal a partir de fragmentos recuperados.
    
    Args:
        fragmentos: Lista de fragmentos recuperados con metadatos
    
    Returns:
        String con el contexto formateado
    """
    textos = []
    for fragmento in fragmentos[:MAX_FRAGMENTOS]:
        textos.append(
            f"[doc={fragmento['doc_id']}, p={fragmento.get('page','?')}, frag={fragmento['frag_id']}]\n"
            f"{fragmento['text']}"
        )
    return "\n\n".join(textos)


def construir_prompt_resumen_parcial(contexto, pregunta):
    """
    Construye el prompt final para el LLM con contexto y pregunta.
    
    Args:
        contexto: Contexto extraido de los fragmentos
        pregunta: Pregunta del usuario
    
    Returns:
        String con el prompt completo formateado
    """
    return (
        "Eres un asistente academico que responde preguntas sobre articulos de investigacion.\n"
        "Instrucciones:\n"
        "1. Responde la pregunta usando SOLO el contexto proporcionado.\n"
        "2. Si el contexto NO contiene suficiente informacion para responder, di:\n"
        "   \"El contexto proporcionado no contiene suficiente informacion para responder la pregunta.\"\n"
        "3. NO uses conocimiento previo.\n"
        "4. NO inventes metodos, resultados o afirmaciones.\n"
        "5. Manten la respuesta corta (2–3 oraciones).\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta:\n{pregunta}\n\n"
        "Respuesta:\n"
    )


# Alias para mantener compatibilidad
MAX_CONTEXT_CHARS = MAX_CARACTERES_CONTEXTO
build_literal_context = construir_contexto_literal
build_partial_summary_prompt = construir_prompt_resumen_parcial
