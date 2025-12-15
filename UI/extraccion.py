# UI/extraccion.py
from pathlib import Path
import sys

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.extract.extractor import extraer_texto_pdf
from src.common.extract.cleaner import limpiar_archivo
from src.common.extract.update_metadata import actualizar_metadata
from src.common.chunking.chunker import generar_chunks
from src.common.embeddings.build_faiss import build_faiss_index

def preprocesar(pdf_path, ui_data_dir="UI/data"):
    """
    Ejecuta la Fase 2 completa:
    - extracción
    - limpieza
    - actualización de metadata
    """
    ui_data_dir = Path(ui_data_dir)

    carpeta_extracted = ui_data_dir / "extracted"
    carpeta_preprocessed = ui_data_dir / "preprocessed"

    carpeta_extracted.mkdir(parents=True, exist_ok=True)
    carpeta_preprocessed.mkdir(parents=True, exist_ok=True)

    pdf_path = Path(pdf_path)
    pdf_id = pdf_path.stem

    # 1️⃣ Extraer texto
    extraer_texto_pdf(pdf_path, carpeta_extracted)

    # 2️⃣ Limpiar texto
    archivo_jsonl = carpeta_extracted / f"{pdf_id}.jsonl"
    limpiar_archivo(archivo_jsonl, carpeta_preprocessed)

    # 3️⃣ Actualizar metadata (cleaned_length, extraction_method, ocr_applied)
    actualizar_metadata(base_data_dir=ui_data_dir)

    generar_chunks(base_data_dir=ui_data_dir)

    build_faiss_index(base_data_dir=ui_data_dir)

    print(f"Fase 4 completada para {pdf_id}")

    
