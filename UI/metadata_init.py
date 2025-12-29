"""
Modulo para inicializar metadatos de PDFs en el CSV.
Extrae informacion basica del documento (numero de paginas, nombre, etc.).
"""

import fitz
import pandas as pd
from pathlib import Path


def inicializar_metadata_pdf(ruta_pdf, ruta_csv_metadata):
    """
    Inicializa o actualiza el CSV de metadatos con informacion de un PDF.
    
    Args:
        ruta_pdf: Ruta al archivo PDF a procesar
        ruta_csv_metadata: Ruta al archivo CSV de metadatos
    """
    ruta_pdf = Path(ruta_pdf)
    documento = fitz.open(ruta_pdf)

    # Crear fila de metadatos
    fila = {
        "filename": ruta_pdf.name,
        "title": ruta_pdf.stem,
        "arxiv_id": "",
        "source_url": "",
        "pages": len(documento),
        "language": "unknown",
        "file_type": "pdf",
        "notes": ""
    }

    # Leer CSV existente o crear nuevo
    if ruta_csv_metadata.exists():
        df = pd.read_csv(ruta_csv_metadata)
        # Verificar si el PDF ya esta registrado
        if fila["filename"] in df["filename"].values:
            return  # Ya registrado
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    else:
        df = pd.DataFrame([fila])

    # Guardar CSV actualizado
    df.to_csv(ruta_csv_metadata, index=False)
