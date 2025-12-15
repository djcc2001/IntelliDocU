"""Construcción de contexto limitado para el LLM."""

from src.v3_rag_advanced.config import MAX_CONTEXT_CHARS, MAX_FRAGMENT_TEXT

def build_limited_context(fragmentos, max_fragments=None, max_chars=None):
    """
    Construye un contexto limitado a partir de los fragmentos.
    
    Args:
        fragmentos: Lista de fragmentos recuperados
        max_fragments: Máximo número de fragmentos (opcional)
        max_chars: Máximo caracteres totales (opcional)
    
    Returns:
        (contexto_str, fragmentos_usados)
    """
    if max_chars is None:
        max_chars = MAX_CONTEXT_CHARS
    
    contexto_partes = []
    usados = []
    chars_acumulados = 0
    
    for frag in fragmentos[:max_fragments]:
        # Extraer y truncar texto si es necesario
        text = frag["text"]
        if len(text) > MAX_FRAGMENT_TEXT:
            text = text[:MAX_FRAGMENT_TEXT] + "..."
        
        # Formatear bloque con metadata
        bloque = (
            f"[Documento: {frag['doc_id']}, "
            f"Página: {frag.get('page', '?')}, "
            f"Sección: {frag.get('section', 'unknown')}]\n"
            f"{text}\n"
        )
        
        # Verificar si excede límite
        if chars_acumulados + len(bloque) > max_chars:
            break
        
        contexto_partes.append(bloque)
        usados.append(frag)
        chars_acumulados += len(bloque)
    
    contexto = "\n---\n".join(contexto_partes)
    return contexto, usados