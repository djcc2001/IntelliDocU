import pandas as pd
import json
from pathlib import Path

def contar_texto_limpio(ruta_jsonl):
    total = 0
    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            registro = json.loads(linea)
            total += len(registro.get("clean_text", ""))
    return total

def actualizar_metadata(base_data_dir="data"):
    base_data_dir = Path(base_data_dir)

    ruta_csv = base_data_dir / "pdf_metadata.csv"
    carpeta_texto = base_data_dir / "preprocessed"

    df = pd.read_csv(ruta_csv)

    if "cleaned_length" not in df.columns:
        df["cleaned_length"] = 0

    if "extraction_method" not in df.columns:
        df["extraction_method"] = "pymupdf"

    if "ocr_applied" not in df.columns:
        df["ocr_applied"] = False

    for i, fila in df.iterrows():
        nombre = fila["filename"].replace(".pdf", "")
        ruta_jsonl = carpeta_texto / f"{nombre}.jsonl"

        if ruta_jsonl.exists():
            largo = contar_texto_limpio(ruta_jsonl)
            df.at[i, "cleaned_length"] = largo

    df.to_csv(ruta_csv, index=False)
    print("Archivo pdf_metadata.csv actualizado")

if __name__ == "__main__":
    actualizar_metadata()
