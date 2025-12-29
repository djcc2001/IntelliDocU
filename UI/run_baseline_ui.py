"""
Modulo UI para ejecutar la version Baseline (V1).
Version sin recuperacion de informacion, solo usa conocimiento interno del LLM.
"""

from pathlib import Path
import sys

# Agregar raiz del proyecto al path
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.common.llm.qwen_llm import ModeloQwen
import random


# =============================
# Configuracion baseline
# =============================
SEMILLA = 42


def construir_prompt(pregunta):
    """
    Construye el prompt para el modelo baseline.
    
    Args:
        pregunta: Pregunta del usuario
    
    Returns:
        Tupla (prompt_sistema, prompt_usuario)
    """
    prompt_sistema = (
        "Eres un asistente academico.\n"
        "Debes responder basandote SOLO en tu conocimiento general.\n"
        "Si no estas seguro de que la informacion sea correcta, "
        "di explicitamente que no lo sabes.\n"
        "NO inventes detalles.\n"
        "NO asumas el contenido de ningun documento especifico.\n"
        "Se conciso y factual."
    )

    prompt_usuario = (
        f"Pregunta:\n{pregunta}\n\n"
        "Respuesta (o indica que la informacion es desconocida):"
    )

    return prompt_sistema, prompt_usuario


def ejecutar_baseline_ui(pregunta: str) -> str:
    """
    Ejecuta baseline v1 sin RAG.
    
    Args:
        pregunta: Pregunta del usuario
    
    Returns:
        Respuesta generada por el modelo
    """
    random.seed(SEMILLA)

    modelo_llm = ModeloQwen()
    prompt = construir_prompt(pregunta)
    respuesta = modelo_llm.generar(prompt).strip()

    return respuesta


# Alias para mantener compatibilidad
run_baseline_ui = ejecutar_baseline_ui
