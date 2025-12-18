# src/v2_rag_basic/prompt.py

MAX_CONTEXT_CHARS = 1200
MAX_FRAGMENTS = 5


def build_literal_context(fragments):
    textos = []
    for f in fragments[:MAX_FRAGMENTS]:
        textos.append(
            f"[doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}]\n"
            f"{f['text']}"
        )
    return "\n\n".join(textos)


def build_partial_summary_prompt(context, question):
    return (
        "You are an academic assistant answering questions about research papers.\n"
        "Instructions:\n"
        "1. Answer the question using ONLY the provided context.\n"
        "2. If the context does NOT contain enough information to answer, say:\n"
        "   \"The provided context does not contain enough information to answer the question.\"\n"
        "3. Do NOT use prior knowledge.\n"
        "4. Do NOT invent methods, results, or claims.\n"
        "5. Keep the answer short (2–3 sentences).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )
