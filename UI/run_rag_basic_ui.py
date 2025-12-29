"""
Modulo UI para ejecutar la version RAG Basico (V2).
Version con recuperacion simple de fragmentos usando FAISS.
"""

from pathlib import Path
import sys
import pickle
import faiss

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.common.llm.qwen_llm import ModeloQwen
from src.v2_rag_basic.rag_pipeline import PipelineRAGBasico

DIRECTORIO_DATOS = "UI/data"

# =============================
# Inicializacion global
# =============================

_modelo_llm = ModeloQwen()


# Cargar FAISS y textos existentes
ruta_indice = Path(DIRECTORIO_DATOS) / "indices/faiss/index.faiss"
ruta_textos = Path(DIRECTORIO_DATOS) / "indices/faiss/texts.pkl"

if ruta_indice.exists() and ruta_textos.exists():
    indice = faiss.read_index(str(ruta_indice))
    with open(ruta_textos, "rb") as archivo:
        textos = pickle.load(archivo)
else:
    indice, textos = None, []

_pipeline_rag = PipelineRAGBasico(_modelo_llm, directorio_base_datos=DIRECTORIO_DATOS)


def ejecutar_rag_basico_ui(pregunta: str) -> str:
    """
    Ejecuta RAG v2 (basico) usando indice FAISS existente.
    Devuelve solo la respuesta para la UI.
    
    Args:
        pregunta: Pregunta del usuario
    
    Returns:
        Respuesta generada por el modelo con contexto
    """
    resultado = _pipeline_rag.responder(pregunta)
    return resultado["answer"]


# Alias para mantener compatibilidad
run_rag_basic_ui = ejecutar_rag_basico_ui
