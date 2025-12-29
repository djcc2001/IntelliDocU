# src/v2_rag_basic/run_rag_eval.py
import json
from pathlib import Path
from src.common.llm.qwen_llm import QwenLLM
#from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v2_rag_basic.rag_pipeline import RAGPipeline

QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v2_rag_basic/rag_basic_answers.json")

def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    llm = QwenLLM()
    rag = RAGPipeline(llm)

    results = []
    for q in questions:
        out = rag.responder(q["question"])
        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": q["question"],
            "type": q["type"],
            "answer": out["answer"],
            "abstained": "not contain enough information" in out["answer"].lower()
        })


    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Resultados de RAG v2 guardados correctamente.")

if __name__ == "__main__":
    main()
