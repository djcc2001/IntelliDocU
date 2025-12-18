# UI/run_rag_advanced_ui.py
from pathlib import Path
import sys
import pickle
import faiss

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.llm.qwen_llm import QwenLLM
from src.v3_rag_advanced.rag_pipeline import RAGAdvancedPipeline

# =============================
# Configuración global
# =============================
DATA_DIR = PROJECT_ROOT / "UI/data"  # ⚡ CAMBIO: path absoluto

# 🔹 Inicializar modelo con offload para memoria limitada
_llm = QwenLLM()
#_llm = QwenLLM(device="auto")

# =============================
# Función para inicializar retriever (⚡ CAMBIO)
# =============================
def init_retriever(base_data_dir: str = str(DATA_DIR)):
    index_path = Path(base_data_dir) / "indices/faiss/index.faiss"
    texts_path = Path(base_data_dir) / "indices/faiss/texts.pkl"

    if index_path.exists() and texts_path.exists():
        index = faiss.read_index(str(index_path))
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)
    else:
        index, texts = None, []

    rag_pipeline = RAGAdvancedPipeline(_llm, base_data_dir=base_data_dir)
    return rag_pipeline

# 🔹 Inicializar pipeline RAG Advanced por defecto

_rag = None  # ⚡ no inicializar al importar

def run_rag_advanced_ui(question: str, retriever=None) -> str:
    global _rag
    if retriever is None:
        if _rag is None:
            # inicializa aquí, cuando se hace la primera pregunta
            _rag = init_retriever()
        retriever = _rag
    result = retriever.answer(question)
    return result["answer"]

