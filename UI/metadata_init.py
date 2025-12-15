import fitz
import pandas as pd
from pathlib import Path

def inicializar_metadata_pdf(pdf_path, metadata_csv):
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)

    fila = {
        "filename": pdf_path.name,
        "title": pdf_path.stem,
        "arxiv_id": "",
        "source_url": "",
        "pages": len(doc),
        "language": "unknown",
        "file_type": "pdf",
        "notes": ""
    }

    if metadata_csv.exists():
        df = pd.read_csv(metadata_csv)
        if fila["filename"] in df["filename"].values:
            return  # ya registrado
        df = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
    else:
        df = pd.DataFrame([fila])

    df.to_csv(metadata_csv, index=False)
