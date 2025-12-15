# src/v2_rag_basic/prompt.py

MAX_CONTEXT_CHARS = 1200
MAX_FRAGMENTS = 5

def build_literal_context(fragments):
    textos = []
    for f in fragments[:MAX_FRAGMENTS]:
        textos.append(
            f"[doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}]\n{f['text']}"
        )
    return "\n\n".join(textos)


def build_partial_summary_prompt(context, question):
    return (
        "You are an assistant helping to explain research papers.\n"
        "Instructions:\n"
        "1. Using ONLY the provided context, answer the question.\n"
        "2. The answer may be approximate or incomplete.\n"
        "3. Do NOT invent new concepts.\n"
        "4. Keep it short (2-3 sentences).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )
