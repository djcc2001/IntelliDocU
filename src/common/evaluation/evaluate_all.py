import json
from metrics import exact_match, f1_score, abstention_accuracy

def load_predictions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(predictions, references):
    em_total = 0
    f1_total = 0
    N = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_answer = pred.get("answer", "").strip().lower()
        ref_answer = ref.get("answer", "").strip().lower()
        em_total += exact_match(pred_answer, ref_answer)
        f1_total += f1_score(pred_answer, ref_answer)

    abst_total = abstention_accuracy(predictions, references)
    return em_total/N, f1_total/N, abst_total

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", required=True, help="Archivo JSON con respuestas correctas")
    parser.add_argument("--preds", nargs="+", required=True, help="Archivos JSON con predicciones a evaluar")
    args = parser.parse_args()

    references = load_predictions(args.refs)

    print(f"{'Modelo':<20} {'Exact Match':<12} {'F1 Score':<10} {'Abstention':<12}")
    print("-" * 60)
    for pred_file in args.preds:
        predictions = load_predictions(pred_file)
        em, f1, abst = evaluate(predictions, references)
        model_name = pred_file.split(".")[0]  # usa el nombre del archivo como etiqueta
        print(f"{model_name:<20} {em:<12.4f} {f1:<10.4f} {abst:<12.4f}")
