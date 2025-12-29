"""
Modulo UI para ejecutar la version RAG Avanzado (V3).
Version con recuperacion avanzada, verificacion de evidencia y citaciones.
"""

from pathlib import Path
import sys
import pickle
import faiss

# Agregar raiz del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.common.llm.qwen_llm import ModeloQwen
from src.v3_rag_advanced.rag_pipeline import PipelineRAGAvanzado

# =============================
# Configuracion global
# =============================
DIRECTORIO_DATOS = RAIZ_PROYECTO / "UI/data"  # Path absoluto

# Inicializar modelo con offload para memoria limitada
_modelo_llm = ModeloQwen()


# =============================
# Funcion para inicializar recuperador
# =============================
def inicializar_recuperador(directorio_base_datos: str = str(DIRECTORIO_DATOS)):
    """
    Inicializa el pipeline RAG avanzado con el recuperador.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Returns:
        Instancia de PipelineRAGAvanzado
    """
    ruta_indice = Path(directorio_base_datos) / "indices/faiss/index.faiss"
    ruta_textos = Path(directorio_base_datos) / "indices/faiss/texts.pkl"

    if ruta_indice.exists() and ruta_textos.exists():
        indice = faiss.read_index(str(ruta_indice))
        with open(ruta_textos, "rb") as archivo:
            textos = pickle.load(archivo)
    else:
        indice, textos = None, []

    pipeline_rag = PipelineRAGAvanzado(_modelo_llm, directorio_base_datos=directorio_base_datos)
    return pipeline_rag


# Pipeline RAG avanzado por defecto (inicializado bajo demanda)
_pipeline_rag = None  # No inicializar al importar


def ejecutar_rag_avanzado_ui(pregunta: str, recuperador=None) -> str:
    """
    Ejecuta RAG v3 (avanzado) con verificacion de evidencia y citaciones.
    
    Args:
        pregunta: Pregunta del usuario
        recuperador: Instancia de PipelineRAGAvanzado (opcional, se crea si no se proporciona)
    
    Returns:
        Respuesta generada por el modelo con contexto y citaciones
    """
    global _pipeline_rag
    if recuperador is None:
        if _pipeline_rag is None:
            # Inicializar aqui, cuando se hace la primera pregunta
            _pipeline_rag = inicializar_recuperador()
        recuperador = _pipeline_rag
    resultado = recuperador.responder(pregunta)
    return resultado["answer"]


# Alias para mantener compatibilidad
init_retriever = inicializar_recuperador
run_rag_advanced_ui = ejecutar_rag_avanzado_ui
