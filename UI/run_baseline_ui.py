# UI/run_baseline_ui.py
from pathlib import Path
import sys

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#from src.common.llm.qwen_llm import QwenLLM
from src.common.llm.flan_t5_llm import FlanT5LLM
import random

# =============================
# Configuración baseline
# =============================
SEED = 42

def build_prompt(question):
    system_prompt = (
        "You are a helpful academic assistant. "
        "Answer the question as clearly and accurately as possible."
    )
    user_prompt = f"Question: {question}\nAnswer:"
    return f"{system_prompt}\n{user_prompt}"

def run_baseline_ui(question: str) -> str:
    """
    Ejecuta baseline v1 sin RAG.
    """
    random.seed(SEED)

    llm = FlanT5LLM()
    prompt = build_prompt(question)
    response = llm.generate(prompt).strip()

    return response
