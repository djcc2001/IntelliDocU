"""Templates de prompts para RAG Advanced."""

from src.v3_rag_advanced.config import ABSTENTION_TEXT

def build_prompt(context, question):
    return (
        "You are an assistant helping to explain research papers.\n"
        "Instructions:\n"
        "1. Answer the question using ONLY the provided context.\n"
        "2. The answer MUST directly address the question and explain the main idea, not copy titles or captions.\n"
        "3. If the question asks for a comparison, explicitly compare the methods mentioned.\n"
        "4. The language of the answer MUST be the same as the language of the question.\n"
        "5. If the context does NOT contain enough information to answer the question, "
        "reply EXACTLY with: \"It is not mentioned in the document.\"\n"
        "6. Do NOT guess, infer, or provide approximate explanations when information is missing.\n"
        "7. If answered, keep the explanation concise (2–3 sentences).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )




def format_answer_with_citations(respuesta, fragmentos):
    """
    Añade citaciones a la respuesta.
    
    Args:
        respuesta: Respuesta generada por el LLM
        fragmentos: Fragmentos usados como contexto
    
    Returns:
        Respuesta con citaciones
    """
    # No añadir citas si es abstención
    if respuesta.strip() == ABSTENTION_TEXT:
        return respuesta
    
    # No añadir citas si no hay fragmentos
    if not fragmentos:
        return respuesta
    
    # Construir citaciones compactas
    citas = []
    for frag in fragmentos:
        cita = (
            f"[Doc: {frag['doc_id']}, "
            f"P.{frag.get('page', '?')}, "
            f"Sec: {frag.get('section', 'unknown')}]"
        )
        citas.append(cita)
    
    # Añadir citas al final
    citas_str = " ".join(citas)
    return f"{respuesta}\n\n📚 Fuentes: {citas_str}"