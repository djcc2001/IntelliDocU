# src/v3_rag_advanced/run_rag_eval.py
import json
from pathlib import Path
from src.common.retriever.retriever import Retriever
from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v3_rag_advanced.abstention import debe_abstener
from src.v3_rag_advanced.context_builder import build_limited_context


# =============================
# Configuración
# =============================
QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v3_rag_advanced/rag_advanced_answers.json")
MAX_CONTEXT_CHARS = 2000         # máximo de caracteres del contexto
TOP_K = 10                       # cantidad de fragmentos a recuperar
MAX_FRAGMENTS = 10               # cantidad máxima de fragmentos a incluir en el contexto
SCORE_MINIMO_ABSTENCION = 0.15  # score mínimo para abstenerse


# =============================
# Funciones de prompt
# =============================
def build_prompt_literal(fragmentos):
    """Extrae literal los fragmentos importantes para mostrar al LLM."""
    textos = []
    for f in fragmentos:
        # Incluimos doc_id, página y frag_id para la generación de citas
        textos.append(f"[doc={f['doc_id']}, p={f.get('page','?')}, frag={f['frag_id']}] {f['text']}")
    return "\n\n".join(textos)

def build_prompt_summary(literal_context, question):
    """
    Prompt ajustado para RAG avanzado con:
    - Grounding estricto
    - Abstención controlada (no binaria)
    - Mejora ligera de F1 sin alucinación
    """
    return (
        "You are an assistant helping to explain academic research papers.\n"
        "Instructions:\n"
        "1. Use ONLY the information explicitly stated in the provided context.\n"
        "2. If the context contains relevant evidence that partially answers the question, "
        "give a cautious and factual explanation limited to that evidence.\n"
        "3. If the context clearly supports a full answer, explain it clearly.\n"
        "4. If the context contains NO relevant information to answer the question, "
        "reply EXACTLY with: \"It is not mentioned in the document.\" and nothing else.\n"
        "5. Do NOT guess, infer beyond the text, or add external knowledge.\n"
        "6. If the question asks for a comparison, explicitly compare ONLY what is stated.\n"
        "7. The language of the answer MUST be the same as the language of the question.\n"
        "8. Keep the answer concise (1–3 sentences).\n\n"
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

def main():
    retriever = Retriever()
    llm = FlanT5LLM()

    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        pregunta = q["question"]

        # 1. Recuperar fragmentos
        fragmentos = retriever.retrieve(pregunta, k=TOP_K)
        # Construir contexto limitado
        contexto, usados = build_limited_context(fragmentos, max_fragments=MAX_FRAGMENTS)

        # 2. Abstención si no hay suficiente info
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
        
        # 4. Guardar resultado
        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": pregunta,
            "type": q["type"],
            "answer": respuesta
        })

    # 5. Guardar todo en archivo JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados de RAG advanced guardados correctamente.")

if __name__ == "__main__":
    main()
