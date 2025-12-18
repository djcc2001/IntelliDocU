import pandas as pd
import json
from pathlib import Path

def contar_texto_limpio(ruta_jsonl):
    total = 0
    paginas = 0

    with open(ruta_jsonl, "r", encoding="utf-8") as f:
        for linea in f:
            registro = json.loads(linea)
            texto = registro.get("clean_text", "")
            total += len(texto)
            paginas += 1

    return total, paginas

def actualizar_metadata(base_data_dir="data"):
    base = Path(base_data_dir)
    ruta_csv = base / "pdf_metadata.csv"
    carpeta = base / "preprocessed"

    df = pd.read_csv(ruta_csv)

    for col, default in {
        "cleaned_length": 0,
        "num_pages_clean": 0,
        "extraction_method": "pymupdf",
        "ocr_applied": False,
    }.items():
        if col not in df.columns:
            df[col] = default

    for i, fila in df.iterrows():
        nombre = fila["filename"].replace(".pdf", "")
        ruta = carpeta / f"{nombre}.jsonl"

        if ruta.exists():
            largo, paginas = contar_texto_limpio(ruta)
            df.at[i, "cleaned_length"] = largo
            df.at[i, "num_pages_clean"] = paginas

    df.to_csv(ruta_csv, index=False)
    print("✓ Metadata actualizada")

if __name__ == "__main__":
    actualizar_metadata()
