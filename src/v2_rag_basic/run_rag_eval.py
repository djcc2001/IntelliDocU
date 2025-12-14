# src/v2_rag_basic/run_rag_eval.py
import json
from pathlib import Path
from src.common.retriever.retriever import Retriever
from src.common.llm.flan_t5_llm import FlanT5LLM

QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v2_rag_basic/rag_basic_answers.json")
TOP_K = 10
MAX_FRAGMENTS = 5 

retriever = Retriever()
llm = FlanT5LLM()

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

def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    for q in questions:
        question_text = q["question"]
        # 1) Recuperar fragmentos
        fragments = retriever.retrieve(question_text, k=TOP_K)
        # 2) Construir contexto literal
        literal_context = build_prompt_literal(fragments)

        # 3) Generar respuesta parcial
        prompt = build_prompt_partial_summary(literal_context, question_text)
        answer = llm.generate(prompt).strip()
        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": q["question"],
            "type": q["type"],
            "answer": answer
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados de RAG basic guardados correctamente.")

if __name__ == "__main__":
    main()
