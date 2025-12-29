"""
Plantillas de prompts para RAG Avanzado.
Define los prompts para generar respuestas y formatear con citaciones.
"""

from src.v3_rag_advanced.config import TEXTO_ABSTENCION


def construir_prompt(contexto, pregunta):
    """
    Construye el prompt principal para el LLM con contexto y pregunta.
    
    Args:
        contexto: Contexto extraido de los fragmentos recuperados
        pregunta: Pregunta del usuario
    
    Returns:
        String con el prompt completo formateado
    """
    return f"""Eres un asistente academico que responde preguntas sobre documentos academicos.

Responde la pregunta usando el contexto proporcionado.

Reglas:
- Usa el contexto para responder la pregunta de manera precisa.
- Si puedes inferir una respuesta razonable del contexto, proporcionala.
- Solo responde EXACTAMENTE con "{TEXTO_ABSTENCION}" si el contexto NO contiene NINGUNA informacion relevante para responder la pregunta.
- Se conciso y factual (2-3 oraciones).

Contexto:
{contexto}

Pregunta: {pregunta}

Respuesta:
"""


def construir_prompt_sin_contexto(pregunta):
    """
    Construye un prompt cuando no hay contexto disponible.
    
    Args:
        pregunta: Pregunta del usuario
    
    Returns:
        String con el prompt de abstención
    """
    return f"""Eres un asistente academico.

NO hay contexto de documento disponible.

Responde EXACTAMENTE con:
"{TEXTO_ABSTENCION}"

Pregunta: {pregunta}
"""


def formatear_respuesta_con_citaciones(respuesta, fragmentos):
    """
    Añade citaciones o disclaimer a la respuesta.
    
    Args:
        respuesta: Respuesta generada por el LLM
        fragmentos: Lista de fragmentos usados como evidencia
    
    Returns:
        Respuesta formateada con citaciones o disclaimer
    """
    if respuesta.strip() == TEXTO_ABSTENCION:
        return respuesta
    
    # Sin fragmentos: añadir disclaimer
    if not fragmentos:
        return f"{respuesta}\n\n Nota: Respuesta basada en conocimiento general, sin fuentes especificas del documento."
    
    # Con fragmentos: añadir citaciones
    citas = []
    vistos = set()
    
    for fragmento in fragmentos:
        paginas = fragmento.get('pages', ['?'])
        # Asegurarnos que sea lista de enteros o strings
        if not isinstance(paginas, list):
            paginas = [paginas]
        paginas_str = ", ".join(str(p) for p in paginas)

        clave_cita = (fragmento['doc_id'], tuple(paginas), fragmento.get('section', 'unknown'))
        if clave_cita not in vistos:
            vistos.add(clave_cita)
            cita = f"[Doc: {fragmento['doc_id']}, Paginas: {paginas_str}, Sec: {fragmento.get('section', 'unknown')}]"
            citas.append(cita)
    
    citas_str = " ".join(citas)
    return f"{respuesta}\n\n📚 Evidencia: {citas_str}"


# Alias para mantener compatibilidad
build_prompt = construir_prompt
build_prompt_without_context = construir_prompt_sin_contexto
format_answer_with_citations = formatear_respuesta_con_citaciones
