# tests/test_ui.py
"""
Test de sanity para los 3 entrypoints de UI.
Puede ejecutarse SIN índices FAISS presentes.
Incluye opción de debug para inspección de errores.
"""

from pathlib import Path
import sys
import traceback

# 🔹 Asegurar root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = "UI/data"
DEBUG = True  # Cambia a False para silencio de errores

def run_test(name, fn, question="What is artificial intelligence?"):
    print(f"\n🧪 TEST: {name}")
    try:
        out = fn(question)
        print("✅ OK")
        if isinstance(out, str):
            print("↪ respuesta:", out[:300])
        else:
            print("↪ tipo:", type(out))
    except Exception as e:
        print("❌ FAILED")
        if DEBUG:
            traceback.print_exc()
        else:
            print("Error:", e)

if __name__ == "__main__":

    # Preguntas de ejemplo
    questions = [
        "¿Qué propiedades fundamentales tiene el problema SSSP?",
        "Según el documento, ¿Qué propiedades fundamentales tiene el problema SSSP?",
        "¿Bajo qué condiciones el nuevo algoritmo O(m log^{2/3} n) supera al algoritmo de Dijkstra?"
    ]

    # 1️⃣ Baseline
    #from UI.run_baseline_ui import ejecutar_baseline_ui
    #for q in questions:
    #    run_test("Baseline UI", ejecutar_baseline_ui, q)

    # 2️⃣ RAG Basic
    #from UI.run_rag_basic_ui import ejecutar_rag_basico_ui
    #for q in questions:
    #    run_test("RAG Basic UI", ejecutar_rag_basico_ui, q)

    # 3️⃣ RAG Advanced
    from UI.run_rag_advanced_ui import ejecutar_rag_avanzado_ui
    for q in questions:
        run_test("RAG Advanced UI", ejecutar_rag_avanzado_ui, q)

    print("\n🎯 Test completo")
