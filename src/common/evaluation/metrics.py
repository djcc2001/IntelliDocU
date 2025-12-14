import json

def exact_match(prediction: str, reference: str) -> int:
    """
    Retorna 1 si la predicción es exactamente igual a la referencia, 0 si no.
    Ignora mayúsculas y espacios extra.
    """
    return int(prediction.strip().lower() == reference.strip().lower())

def f1_score(prediction: str, reference: str) -> float:
    """
    Calcula F1 a nivel de tokens entre la predicción y la referencia.
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def abstention_accuracy(predictions: list, references: list, abstention_token="It is not mentioned in the document.") -> float:
    """
    Precisión de abstención: cuántas veces el sistema se abstuvo correctamente.
    """
    correct = 0
    total = len(predictions)
    for pred, ref in zip(predictions, references):
        pred_answer = pred.get("answer", "").strip().lower()
        ref_answer = ref.get("answer", "").strip().lower()
        if ref_answer == abstention_token.lower() and pred_answer == abstention_token.lower():
            correct += 1
    return correct / total

