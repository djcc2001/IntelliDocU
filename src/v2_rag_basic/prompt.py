# src/v2_rag_basic/prompt.py

def build_prompt(question, fragments):
    contexto = ""

    for i, frag in enumerate(fragments):
        contexto += (
            f"[FRAGMENT {i+1}] "
            f"(doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n"
            f"{frag['text']}\n\n"
        )

    prompt = f"""
You are an assistant that answers questions using ONLY the provided context.
If the answer is not contained in the context, say you do not know.

Context:
{contexto}

Question:
{question}

Answer:
"""
    return prompt
