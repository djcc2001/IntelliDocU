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

def abstention_accuracy(predictions: list, references: list, abstention_token="No se menciona en el documento.") -> float:
    """
    Precisión de abstención: cuántas veces el sistema se abstuvo correctamente.
    
    Args:
        predictions: Lista de predicciones del sistema
        references: Lista de respuestas de referencia
        abstention_token: Token de abstención (por defecto en español)
    
    Returns:
        Porcentaje de abstenciones correctas
    """
    correct = 0
    total = len(predictions)
    
    # Tokens de abstención posibles (inglés y español para compatibilidad)
    tokens_abstencion = [
        abstention_token.lower(),
        "it is not mentioned in the document.".lower(),  # Compatibilidad con datos antiguos
        "no se menciona en el documento.".lower()
    ]
    
    for pred, ref in zip(predictions, references):
        pred_answer = pred.get("answer", "").strip().lower()
        ref_answer = ref.get("answer", "").strip().lower()
        
        # Verificar si ambos son tokens de abstención (cualquiera de los posibles)
        pred_abstiene = pred_answer in tokens_abstencion
        ref_abstiene = ref_answer in tokens_abstencion
        
        if ref_abstiene and pred_abstiene:
            correct += 1
    
    return correct / total if total > 0 else 0.0

