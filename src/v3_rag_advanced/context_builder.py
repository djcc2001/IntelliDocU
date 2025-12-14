# src/v3_rag_advanced/context_builder.py
MAX_CONTEXT_CHARS = 1200
MAX_FRAGMENT_TEXT = 400  # truncar fragmentos largos

def build_limited_context(fragmentos, max_fragments=2):
    contexto = ""
    usados = []

    for i, frag in enumerate(fragmentos[:max_fragments]):
        text = frag["text"]
        if len(text) > MAX_FRAGMENT_TEXT:
            text = text[:MAX_FRAGMENT_TEXT] + " ..."

        bloque = f"(doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n{text}\n\n"
        if len(contexto) + len(bloque) > MAX_CONTEXT_CHARS:
            break

        contexto += bloque
        usados.append(frag)

    return contexto, usados
