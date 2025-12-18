# UI/run_rag_basic_ui.py
from pathlib import Path
import sys
import pickle
import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.llm.qwen_llm import QwenLLM
from src.v2_rag_basic.rag_pipeline import RAGPipeline

DATA_DIR = "UI/data"

# =============================
# Inicialización global
# =============================

_llm = QwenLLM()


# 🔹 Cargar FAISS y textos existentes
index_path = Path(DATA_DIR) / "indices/faiss/index.faiss"
texts_path = Path(DATA_DIR) / "indices/faiss/texts.pkl"

if index_path.exists() and texts_path.exists():
    index = faiss.read_index(str(index_path))
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
else:
    index, texts = None, []

_rag = RAGPipeline(_llm, base_data_dir=DATA_DIR)

def run_rag_basic_ui(question: str) -> str:
    """
    Ejecuta RAG v2 (basic) usando índice FAISS existente.
    Devuelve solo la respuesta para la UI.
    """
    result = _rag.answer(question)
    return result["answer"]
