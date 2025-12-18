# UI/run_baseline_ui.py
from pathlib import Path
import sys

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.llm.qwen_llm import QwenLLM
#from src.common.llm.flan_t5_llm import FlanT5LLM
import random

# =============================
# Configuración baseline
# =============================
SEED = 42

def build_prompt(question):
    system_prompt = (
        "You are an academic assistant.\n"
        "You must answer based ONLY on your general knowledge.\n"
        "If you are not certain that the information is correct, "
        "explicitly say that you do not know.\n"
        "Do NOT invent details.\n"
        "Do NOT assume the contents of any specific document.\n"
        "Be concise and factual."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        "Answer (or state that the information is unknown):"
    )

    return system_prompt, user_prompt

def run_baseline_ui(question: str) -> str:
    """
    Ejecuta baseline v1 sin RAG.
    """
    random.seed(SEED)

    llm = QwenLLM()
    prompt = build_prompt(question)
    response = llm.generate(prompt).strip()

    return response
