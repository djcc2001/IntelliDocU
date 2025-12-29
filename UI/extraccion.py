"""
Modulo para preprocesar PDFs en la UI.
Ejecuta el pipeline completo: extraccion, limpieza, fragmentacion e indexacion.
"""

from pathlib import Path
import sys

# Agregar raiz del proyecto
RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from src.common.extract.extractor import extraer_texto_pdf
from src.common.extract.cleaner import limpiar_archivo
from src.common.extract.update_metadata import actualizar_metadata
from src.common.chunking.chunker import generar_chunks
from src.common.embeddings.build_faiss import construir_indice_faiss


def preprocesar(ruta_pdf, directorio_datos_ui="UI/data"):
    """
    Ejecuta el pipeline completo de preprocesamiento:
    - Extraccion de texto
    - Limpieza y normalizacion
    - Actualizacion de metadatos
    - Fragmentacion (chunking)
    - Construccion de indice FAISS
    
    Args:
        ruta_pdf: Ruta al archivo PDF a procesar
        directorio_datos_ui: Directorio donde se guardaran los datos procesados
    """
    directorio_datos_ui = Path(directorio_datos_ui)

    carpeta_extraidos = directorio_datos_ui / "extracted"
    carpeta_preprocesados = directorio_datos_ui / "preprocessed"

    carpeta_extraidos.mkdir(parents=True, exist_ok=True)
    carpeta_preprocesados.mkdir(parents=True, exist_ok=True)

    ruta_pdf = Path(ruta_pdf)
    identificador_pdf = ruta_pdf.stem

    # 1️⃣ Extraer texto
    extraer_texto_pdf(ruta_pdf, carpeta_extraidos)

    # 2️⃣ Limpiar texto
    archivo_jsonl = carpeta_extraidos / f"{identificador_pdf}.jsonl"
    limpiar_archivo(archivo_jsonl, carpeta_preprocesados)

    # 3️⃣ Actualizar metadatos (cleaned_length, extraction_method, ocr_applied)
    actualizar_metadata(directorio_base_datos=directorio_datos_ui)

    # 4️⃣ Generar fragmentos
    generar_chunks(directorio_base_datos=directorio_datos_ui)

    # 5️⃣ Construir indice FAISS
    construir_indice_faiss(directorio_base_datos=directorio_datos_ui)

    print(f"Fase 4 completada para {identificador_pdf}")
