# UI/run_rag_basic_ui.py
from pathlib import Path
import sys

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#from src.common.llm.qwen_llm import QwenLLM
from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v2_rag_basic.rag_pipeline import RAGPipeline

# =============================
# Inicialización global
# =============================
_llm = FlanT5LLM()
_rag = RAGPipeline(_llm)

def run_rag_basic_ui(question: str) -> str:
    """
    Ejecuta RAG v2 (basic).
    Devuelve solo la respuesta para la UI.
    """
    result = _rag.answer(question)
    return result["answer"]

