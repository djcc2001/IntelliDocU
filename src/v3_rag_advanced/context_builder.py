"""
Construccion de contexto limitado para el LLM.
Agrupa y formatea fragmentos recuperados para incluir en el prompt.
"""

from src.v3_rag_advanced.config import MAX_CARACTERES_CONTEXTO, MAX_TEXTO_FRAGMENTO


def construir_contexto_limitatado(fragmentos, max_fragmentos=5):
    """
    Construye un contexto limitado a partir de los fragmentos recuperados.
    
    Args:
        fragmentos: Lista de fragmentos recuperados
        max_fragmentos: Maximo numero de fragmentos a incluir
    
    Returns:
        Tupla (contexto_str, fragmentos_usados)
    """
    fragmentos_usados = []
    partes_contexto = []

    for fragmento in fragmentos:
        texto = fragmento.get("text", "").strip()
        if not texto:
            continue

        fragmentos_usados.append(fragmento)
        partes_contexto.append(
            f"[{len(fragmentos_usados)}] {texto}"
        )

        if len(fragmentos_usados) >= max_fragmentos:
            break

    contexto = "\n\n".join(partes_contexto)
    return contexto, fragmentos_usados


# Alias para mantener compatibilidad
build_limited_context = construir_contexto_limitatado
