import json
from pathlib import Path

from src.common.retriever.retriever import Retriever
from src.common.llm.simple_llm import SimpleLLM


# =============================
# Configuracion
# =============================
QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v3_rag_advanced/rag_advanced_answers.json")

K = 5
MIN_FRAGMENTS = 1


# =============================
# Inicializacion
# =============================
retriever = Retriever()
llm = SimpleLLM()


def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        question_text = q["question"]

        fragments = retriever.retrieve(question_text, k=K)

        # =============================
        # Regla de abstencion
        # =============================
        if len(fragments) < MIN_FRAGMENTS:
            answer = "No se encontro informacion suficiente en el PDF para responder esta pregunta."
        else:
            context = ""
            for i, frag in enumerate(fragments, start=1):
                context += (
                    f"[FRAGMENT {i}] "
                    f"(doc:{frag['doc_id']}, page:{frag['page']}, frag:{frag['frag_id']}):\n"
                    f"{frag['text']}\n\n"
                )

            prompt = f"""
You are an assistant that answers questions using ONLY the provided context.
Each factual statement MUST include a citation in the format:
[cita: doc=DOC_ID, p=PAGE, frag=FRAG_ID]

If the answer is not fully supported by the context, respond with:
No se encontro informacion suficiente en el PDF para responder esta pregunta.

Context:
{context}

Question:
{question_text}

Answer:
"""
            answer = llm.generate(prompt).strip()

            # Seguridad extra: si el modelo ignora las reglas
            if "cita:" not in answer:
                answer = "No se encontro informacion suficiente en el PDF para responder esta pregunta."

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

    print("Resultados de RAG advanced guardados correctamente.")


if __name__ == "__main__":
    main()
