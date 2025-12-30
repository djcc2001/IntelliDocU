"""
Evaluacion completa del sistema Baseline (V1) sobre el dataset de preguntas.
Esta version no utiliza recuperacion de informacion, solo conocimiento interno del modelo.
"""

import json
from pathlib import Path
from tqdm import tqdm
from src.v1_baseline.run_baseline import run_baseline

# Rutas
RUTA_PREGUNTAS = Path("data/questions/questions.json")
DIRECTORIO_RESULTADOS = Path("results/v1_baseline")
ARCHIVO_SALIDA = DIRECTORIO_RESULTADOS / "baseline_answers.json"


def evaluar():
    """
    Evalua el sistema Baseline sobre todas las preguntas del dataset.
    """
    # Cargar preguntas
    print("Cargando preguntas...")
    with open(RUTA_PREGUNTAS, "r", encoding="utf-8") as archivo:
        preguntas = json.load(archivo)

    print(f"Total de preguntas: {len(preguntas)}")

    # Inicializar sistema
    print("Inicializando Baseline (V1)...")
    print("Nota: Esta version no utiliza recuperacion de informacion.")

    # Procesar preguntas
    resultados = []
    
    for pregunta in tqdm(preguntas, desc="Evaluando"):
        # Usar la funcion run_baseline de run_baseline.py
        respuesta = run_baseline(pregunta["question"])
        
        resultados.append({
            "question_id": pregunta["id"],
            "doc_id": pregunta["doc_id"],
            "question": pregunta["question"],
            "type": pregunta["type"],
            "answer": respuesta,
            "abstained": False  # Baseline no tiene mecanismo de abstención
        })

    # Guardar resultados
    DIRECTORIO_RESULTADOS.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVO_SALIDA, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2, ensure_ascii=False)

    # Estadisticas
    print(f"\n✓ Evaluacion completada")
    print(f"  Resultados guardados en: {ARCHIVO_SALIDA}")
    print(f"  Total de preguntas procesadas: {len(resultados)}")
    print(f"  Nota: Baseline no se abstiene, responde todas las preguntas.")


if __name__ == "__main__":
    evaluar()
