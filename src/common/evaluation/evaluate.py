import json
from metrics import exact_match, f1_score, abstention_accuracy

def load_predictions(file_path):
    """Carga predicciones y respuestas de un archivo JSON"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(pred_file, reference_file):
    predictions = load_predictions(pred_file)
    references = load_predictions(reference_file)

    em_total = 0
    f1_total = 0
    abst_total = 0
    N = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_answer = pred.get("answer", "")
        ref_answer = ref.get("answer", "")
        em_total += exact_match(pred_answer, ref_answer)
    f1_total += f1_score(pred_answer, ref_answer)

    abst_total = abstention_accuracy(predictions, references)

    print(f"Exact Match: {em_total/N:.4f}")
    print(f"F1 Score: {f1_total/N:.4f}")
    print(f"Abstention Accuracy: {abst_total:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True, help="Archivo JSON con predicciones")
    parser.add_argument("--reference_file", required=True, help="Archivo JSON con respuestas correctas")
    args = parser.parse_args()

    evaluate(args.pred_file, args.reference_file)
