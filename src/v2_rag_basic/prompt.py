MAX_CONTEXT_CHARS = 1200  # aumentar ligeramente para Flan-T5 (~300 tokens)

def build_prompt(question, fragments):
    contexto = ""
    for i, frag in enumerate(fragments):
        fragment_text = (
            f"[FRAGMENT {i+1}] (doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n"
            f"{frag['text']}\n\n"
        )
        if len(contexto) + len(fragment_text) > MAX_CONTEXT_CHARS:
            break
        contexto += fragment_text

    prompt = (
        "You are an assistant that answers questions using ONLY the provided context.\n"
        "If the question asks 'what is', provide a concise definition.\n"
        "If the question asks about methods/results, summarize based on context.\n"
        "If the answer is not in the context, respond exactly: 'It is not mentioned in the document.'\n\n"
        f"Context:\n{contexto}\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )
    return prompt
