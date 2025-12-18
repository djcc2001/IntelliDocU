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
DATA_DIR = "UI/data"

# 🔹 Inicializar modelo con offload para memoria limitada
_llm = QwenLLM(device="auto")  # ya no da error de argumento

# 🔹 Cargar FAISS y textos existentes
index_path = Path(DATA_DIR) / "indices/faiss/index.faiss"
texts_path = Path(DATA_DIR) / "indices/faiss/texts.pkl"

if index_path.exists() and texts_path.exists():
    index = faiss.read_index(str(index_path))
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)
else:
    index, texts = None, []

# 🔹 Inicializar pipeline RAG Advanced
_rag = RAGAdvancedPipeline(_llm, base_data_dir=DATA_DIR)

def run_rag_advanced_ui(question: str) -> str:
    """
    Ejecuta RAG v3 (advanced) usando índice FAISS existente.
    Maneja GPU con memoria limitada.
    """
    result = _rag.answer(question)
    return result["answer"]
