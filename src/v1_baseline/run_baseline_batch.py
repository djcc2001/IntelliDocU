import json
from pathlib import Path
from run_baseline import run_baseline

QUESTIONS_PATH = Path("data/questions/questions.json")
OUTPUT_PATH = Path("results/v1_baseline/baseline_answers.json")

def main():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for q in questions:
        answer = run_baseline(q["question"])

        results.append({
            "question_id": q["id"],
            "doc_id": q["doc_id"],
            "question": q["question"],
            "type": q["type"],
            "answer": answer
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Resultados del baseline guardados correctamente.")

if __name__ == "__main__":
    main()
