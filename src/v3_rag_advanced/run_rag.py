# src/v3_rag_advanced/run_rag.py

from src.common.retriever.retriever import Retriever
from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v3_rag_advanced.abstention import debe_abstener
from src.v3_rag_advanced.context_builder import build_limited_context

# =============================
# Configuración
# =============================
MAX_CONTEXT_CHARS = 2000         # máximo de caracteres del contexto
TOP_K = 10                       # cantidad de fragmentos a recuperar
MAX_FRAGMENTS = 10               # cantidad máxima de fragmentos a incluir en el contexto
SCORE_MINIMO_ABSTENCION = 0.15  # score mínimo para abstenerse

# =============================
# Funciones de prompt
# =============================
def build_prompt_literal(fragmentos):
    textos = []
    for f in fragmentos:
        textos.append(f"[doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}] {f['text']}")
    return "\n\n".join(textos)

def build_prompt_summary(literal_context, question):
    """
    Prompt con control de idioma, comparación explícita y abstención estricta.
    """
    return (
        "You are an assistant helping to explain research papers.\n"
        "Instructions:\n"
        "1. Answer the question using ONLY the provided context.\n"
        "2. The answer MUST directly address the question and explain the main idea, not copy titles or captions.\n"
        "3. If the question asks for a comparison, explicitly compare the methods mentioned.\n"
        "4. The language of the answer MUST be the same as the language of the question.\n"
        "5. If the context does NOT contain enough information to answer the question, "
        "reply EXACTLY with: \"It is not mentioned in the document.\"\n"
        "6. Do NOT guess, infer, or provide approximate explanations when information is missing.\n"
        "7. If answered, keep the explanation concise (2–3 sentences).\n\n"
        f"Context:\n{literal_context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )



def agregar_citas(respuesta, fragmentos_usados):
    """Agrega todas las citas de los fragmentos usados al final de la respuesta."""
    if respuesta == "It is not mentioned in the document.":
        return respuesta  # No agregamos citas si es abstención
    citas = []
    for f in fragmentos_usados:
        citas.append(f"[cita: doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}]")
    if citas:
        return f"{respuesta} {' '.join(citas)}"
    return respuesta

# =============================
# Ejecución principal
# =============================
def main():
    pregunta = "Does the paper propose using a quantum neural network for training?"
    retriever = Retriever()
    llm = FlanT5LLM()

    # Recuperar fragmentos
    fragmentos = retriever.retrieve(pregunta, k=TOP_K)

    # Construir contexto limitado
    contexto, usados = build_limited_context(fragmentos, max_fragments=MAX_FRAGMENTS)

    # Abstención si no hay suficiente info
    if debe_abstener(pregunta, usados, score_minimo=SCORE_MINIMO_ABSTENCION):
        respuesta = "It is not mentioned in the document."
    else:
        # 1) Extraer literalmente los fragmentos
        literal_context = build_prompt_literal(usados)

        # 2) Generar explicación resumida basada solo en esos fragmentos
        prompt = build_prompt_summary(literal_context, pregunta)
        respuesta = llm.generate(prompt).strip()

        # 3) Agregar citas de todos los fragmentos usados
        respuesta = agregar_citas(respuesta, usados)

    print("\n=== RAG HYBRID RESPONSE ===\n")
    print(respuesta)

# =============================
# Entrada del script
# =============================
if __name__ == "__main__":
    main()
