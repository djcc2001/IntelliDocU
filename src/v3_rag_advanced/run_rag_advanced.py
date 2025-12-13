# src/v3_rag_advanced/run_rag_advanced.py

from src.common.retriever.retriever import Retriever
from src.common.llm.simple_llm import SimpleLLM
from src.v3_rag_advanced.abstention import debe_abstener


# -------------------------------
# Construccion del prompt avanzado
# -------------------------------

def construir_prompt_avanzado(pregunta, fragmentos):
    """
    Construye un prompt que obliga al modelo a citar evidencia
    """
    instrucciones = (
        "You are an assistant that answers questions using ONLY the provided context.\n"
        "Each factual statement MUST include a citation in the format:\n"
        "[cita: doc=DOC_ID, p=PAGE, frag=FRAG_ID]\n"
        "If the answer is not contained in the context, respond with:\n"
        "No se encontro informacion suficiente en el PDF para responder esta pregunta.\n\n"
    )

    contexto = "Context:\n"
    for i, frag in enumerate(fragmentos, 1):
        contexto += (
            f"[FRAGMENT {i}] "
            f"(doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n"
            f"{frag['text']}\n\n"
        )

    prompt = (
        instrucciones
        + contexto
        + f"Question:\n{pregunta}\n\n"
        + "Answer:\n"
    )

    return prompt


# -------------------------------
# Ejecucion principal
# -------------------------------

def main():
    pregunta = "¿Hola como estás?"

    retriever = Retriever()
    llm = SimpleLLM()

    # Recuperar fragmentos
    fragmentos = retriever.retrieve(pregunta, k=5)
 
    if debe_abstener(pregunta, fragmentos):
        print("No se encontro informacion suficiente en el PDF para responder esta pregunta.")
        return

    prompt = construir_prompt_avanzado(pregunta, fragmentos)

    respuesta = llm.generate(prompt)

    print("\n=== RAG ADVANCED RESPONSE (RAW) ===\n")
    print(respuesta)


if __name__ == "__main__":
    main()
