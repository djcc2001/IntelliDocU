"""Templates de prompts para RAG Advanced."""

from src.v3_rag_advanced.config import ABSTENTION_TEXT


def build_prompt(context, question):
    return f"""You are a careful academic assistant.

Answer the question using ONLY the provided context.

Rules:
- The answer MUST be directly supported by at least one sentence in the context.
- Do NOT infer, generalize, or use external knowledge.
- If the answer is NOT explicitly stated, respond EXACTLY with:
"{ABSTENTION_TEXT}"
- Be concise and factual (max 2 sentences).

Context:
{context}

Question: {question}

Answer:
"""



def build_prompt_without_context(question):
    return f"""You are an academic assistant.

There is NO document context available.

Respond EXACTLY with:
"{ABSTENTION_TEXT}"

Question: {question}
"""



def format_answer_with_citations(respuesta, fragmentos):
    """Añade citaciones o disclaimer."""
    if respuesta.strip() == ABSTENTION_TEXT:
        return respuesta
    
    # Sin fragmentos: añadir disclaimer
    if not fragmentos:
        return f"{respuesta}\n\n Nota: Respuesta basada en conocimiento general, sin fuentes específicas del documento."
    
    # Con fragmentos: añadir citaciones
    citas = []
    seen = set()
    
    for frag in fragmentos:
        cita_key = (frag['doc_id'], frag.get('page', '?'), frag.get('section', 'unknown'))
        if cita_key not in seen:
            seen.add(cita_key)
            cita = f"[Doc: {frag['doc_id']}, P.{frag.get('page', '?')}, Sec: {frag.get('section', 'unknown')}]"
            citas.append(cita)
    
    citas_str = " ".join(citas)
    return f"{respuesta}\n\n📚 Evidence: {citas_str}"
