# src/v2_rag/run_rag_v2_partial.py

from src.common.retriever.retriever import Retriever
from src.common.llm.flan_t5_llm import FlanT5LLM

TOP_K = 10
MAX_FRAGMENTS = 5  # cuantos fragmentos usamos para el contexto

# =============================
# Funciones de prompt
# =============================
def build_prompt_literal(fragmentos):
    """Concatena fragmentos con información de doc_id, page y frag_id"""
    textos = []
    for f in fragmentos[:MAX_FRAGMENTS]:
        textos.append(f"[doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}] {f['text']}")
    return "\n\n".join(textos)

def build_prompt_partial_summary(literal_context, question):
    """
    Prompt para generar explicación entendible pero parcialmente correcta.
    La respuesta puede contener errores o ser incompleta.
    """
    return (
        "You are an assistant helping to explain AI research papers.\n"
        "Instructions:\n"
        "1. Using ONLY the provided context, give a readable explanation answering the question.\n"
        "2. The explanation can be approximate, may contain minor errors or incomplete info.\n"
        "3. Do NOT invent entirely new concepts.\n"
        "4. Keep the answer short and understandable, 2-3 sentences.\n\n"
        f"Context:\n{literal_context}\n\n"
        f"Question:\n{question}\n\n"
        "Partial Answer:\n"
    )

# =============================
# Ejecución principal
# =============================
def main():
    pregunta = "How does SPARSESWAPS improve pruning for LLMs compared to traditional magnitude pruning methods?"
    retriever = Retriever()
    llm = FlanT5LLM()

    # 1) Recuperar fragmentos
    fragmentos = retriever.retrieve(pregunta, k=TOP_K)

    # 2) Construir contexto literal
    literal_context = build_prompt_literal(fragmentos)

    # 3) Generar respuesta parcial
    prompt = build_prompt_partial_summary(literal_context, pregunta)
    respuesta = llm.generate(prompt).strip()

    print("\n=== RAG V2 PARTIAL RESPONSE ===\n")
    print(respuesta)


if __name__ == "__main__":
    main()
