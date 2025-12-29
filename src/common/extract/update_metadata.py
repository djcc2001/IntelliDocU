"""
Modulo para actualizar metadatos de PDFs con informacion del procesamiento.
Calcula estadisticas como longitud de texto limpio y numero de paginas procesadas.
"""

import pandas as pd
import json
from pathlib import Path


def contar_texto_limpio(ruta_jsonl):
    """
    Cuenta el total de caracteres y paginas en un archivo JSONL preprocesado.
    
    Args:
        ruta_jsonl: Ruta al archivo JSONL con texto preprocesado
    
    Returns:
        Tupla (total_caracteres, numero_paginas)
    """
    total_caracteres = 0
    numero_paginas = 0

    with open(ruta_jsonl, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            registro = json.loads(linea)
            texto = registro.get("clean_text", "")
            total_caracteres += len(texto)
            numero_paginas += 1

    return total_caracteres, numero_paginas


def actualizar_metadata(directorio_base_datos="data"):
    """
    Actualiza el archivo CSV de metadatos con estadisticas del procesamiento.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Actualiza las columnas:
    - cleaned_length: Total de caracteres en texto limpio
    - num_pages_clean: Numero de paginas procesadas
    - extraction_method: Metodo de extraccion usado (pymupdf)
    - ocr_applied: Si se aplico OCR (False por defecto)
    """
    base = Path(directorio_base_datos)
    ruta_csv = base / "pdf_metadata.csv"
    carpeta_preprocesados = base / "preprocessed"

    # Leer CSV existente
    df = pd.read_csv(ruta_csv)

    # Agregar columnas si no existen
    for columna, valor_defecto in {
        "cleaned_length": 0,
        "num_pages_clean": 0,
        "extraction_method": "pymupdf",
        "ocr_applied": False,
    }.items():
        if columna not in df.columns:
            df[columna] = valor_defecto

    # Actualizar estadisticas para cada PDF
    for indice, fila in df.iterrows():
        nombre_sin_extension = fila["filename"].replace(".pdf", "")
        ruta_jsonl = carpeta_preprocesados / f"{nombre_sin_extension}.jsonl"

        if ruta_jsonl.exists():
            largo_texto, paginas = contar_texto_limpio(ruta_jsonl)
            df.at[indice, "cleaned_length"] = largo_texto
            df.at[indice, "num_pages_clean"] = paginas

    # Guardar CSV actualizado
    df.to_csv(ruta_csv, index=False)
    print("✓ Metadatos actualizados")


if __name__ == "__main__":
    actualizar_metadata()
