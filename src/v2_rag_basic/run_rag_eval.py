import json
from pathlib import Path

from src.common.retriever.retriever import Retriever
from src.common.llm.simple_llm import SimpleLLM


# =============================
# Configuracion
# =============================
QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v2_rag_basic/rag_basic_answers.json")

K = 5


# =============================
# Inicializacion
# =============================
retriever = Retriever()
llm = SimpleLLM()


def main():
    # Cargar preguntas
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    # Loop principal (igual que baseline)
    for q in questions:
        question_text = q["question"]

        # Recuperacion
        fragments = retriever.retrieve(question_text, k=K)

        # Construccion del contexto
        context = ""
        for i, frag in enumerate(fragments, start=1):
            context += (
                f"[FRAGMENT {i}] "
                f"(doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n"
                f"{frag['text']}\n\n"
            )

        # Prompt RAG basico
        prompt = f"""
You are an assistant that answers questions using ONLY the provided context.
If the answer is not contained in the context, say you do not know.

Context:
{context}

Question:
{question_text}

Answer:
"""

        answer = llm.generate(prompt)

        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": q["question"],
            "type": q["type"],
            "answer": answer.strip()
        })

    # Guardar resultados
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados de RAG basic guardados correctamente.")


if __name__ == "__main__":
    main()
