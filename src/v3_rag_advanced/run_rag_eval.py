"""Evaluación completa sobre el dataset de preguntas (pipeline ajustado)."""

import json
from pathlib import Path
from tqdm import tqdm
from src.common.llm.qwen_llm import QwenLLM
#from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v3_rag_advanced.rag_pipeline import RAGAdvancedPipeline
from src.v3_rag_advanced.config import ABSTENTION_TEXT

# Rutas
QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_DIR = Path("results/v3_rag_advanced")
OUTPUT_FILE = OUTPUT_DIR / "rag_advanced_answers.json"


def evaluate():
    """Evalúa el sistema RAG Advanced sobre todas las preguntas."""

    # Cargar preguntas
    print("Cargando preguntas...")
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Total de preguntas: {len(questions)}")

    # Inicializar sistema
    print("Inicializando RAG Advanced...")
    llm = QwenLLM()
    rag = RAGAdvancedPipeline(llm)

    # Procesar preguntas
    results = []
    abstenciones = 0
    posibles_hallucinations = 0

    for q in tqdm(questions, desc="Evaluando"):
        result = rag.answer(q["question"])

        is_abstain = result["answer"] == ABSTENTION_TEXT
        if is_abstain:
            abstenciones += 1
        elif not result["fragments"]:
            # No se abstuvo pero no hay fragmentos → posible hallucination
            posibles_hallucinations += 1

        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": q["question"],
            "type": q["type"],
            "answer": result["answer"],
            "abstained": is_abstain,
            "num_fragments": len(result["fragments"])
        })

    # Guardar resultados
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Estadísticas
    print(f"\n✓ Evaluación completada")
    print(f"  Resultados guardados en: {OUTPUT_FILE}")
    print(f"\nEstadísticas generales:")
    print(f"  - Total de preguntas: {len(results)}")
    print(f"  - Respuestas generadas: {len(results) - abstenciones}")
    print(f"  - Abstenciones: {abstenciones}")
    print(f"  - Posibles hallucinations: {posibles_hallucinations}")
    print(f"  - Tasa de abstención: {abstenciones/len(results)*100:.1f}%")

    # Estadísticas por tipo
    factual = [r for r in results if r["type"] == "factual"]
    impossible = [r for r in results if r["type"] == "impossible"]

    if factual:
        abs_fact = sum(1 for r in factual if r["abstained"])
        print(f"\n  Preguntas factuales:")
        print(f"    - Total: {len(factual)}")
        print(f"    - Abstenciones: {abs_fact}")

    if impossible:
        abs_imp = sum(1 for r in impossible if r["abstained"])
        print(f"\n  Preguntas imposibles:")
        print(f"    - Total: {len(impossible)}")
        print(f"    - Abstenciones: {abs_imp} (ideal: {len(impossible)})")


if __name__ == "__main__":
    evaluate()
