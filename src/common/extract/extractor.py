"""
Modulo de extraccion de texto desde archivos PDF.
Utiliza PyMuPDF (fitz) para extraer texto de cada pagina del documento.
"""

import fitz  # PyMuPDF
import json
from pathlib import Path


def extraer_texto_pdf(ruta_pdf, carpeta_salida):
    """
    Extrae el texto de todas las paginas de un archivo PDF.
    
    Args:
        ruta_pdf: Ruta al archivo PDF a procesar
        carpeta_salida: Directorio donde se guardara el archivo JSONL con el texto extraido
    
    Genera un archivo JSONL con una linea por pagina, conteniendo:
    - pdf_id: Identificador del documento (nombre sin extension)
    - page: Numero de pagina
    - text: Texto extraido de la pagina
    """
    documento = fitz.open(ruta_pdf)

    identificador_pdf = Path(ruta_pdf).stem
    ruta_salida = Path(carpeta_salida) / f"{identificador_pdf}.jsonl"

    with open(ruta_salida, "w", encoding="utf-8") as archivo_salida:
        for numero_pagina, pagina in enumerate(documento, start=1):
            bloques = pagina.get_text("blocks")

            # Extraer texto de todos los bloques de texto de la pagina
            texto = "\n".join(
                bloque[4].strip()
                for bloque in bloques
                if len(bloque) > 4 and bloque[4].strip()
            )

            # Ignorar paginas casi vacias (menos de 30 caracteres)
            if len(texto) < 30:
                continue

            registro = {
                "pdf_id": identificador_pdf,
                "page": numero_pagina,
                "text": texto
            }

            archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"✓ Texto extraido: {identificador_pdf}")


if __name__ == "__main__":
    # Configuracion de rutas por defecto
    carpeta_pdfs = "data/pdfs"
    carpeta_salida = "data/extracted"

    # Crear directorio de salida si no existe
    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    # Procesar todos los PDFs en la carpeta
    for pdf in Path(carpeta_pdfs).glob("*.pdf"):
        extraer_texto_pdf(pdf, carpeta_salida)
