import fitz  # PyMuPDF
import json
from pathlib import Path

def extraer_texto_pdf(ruta_pdf, carpeta_salida):
    doc = fitz.open(ruta_pdf)

    pdf_id = Path(ruta_pdf).stem
    ruta_salida = Path(carpeta_salida) / f"{pdf_id}.jsonl"

    with open(ruta_salida, "w", encoding="utf-8") as f:
        for num_pagina, pagina in enumerate(doc, start=1):
            bloques = pagina.get_text("blocks")

            texto = "\n".join(
                b[4].strip()
                for b in bloques
                if len(b) > 4 and b[4].strip()
            )

            # Ignorar páginas casi vacías
            if len(texto) < 30:
                continue

            registro = {
                "pdf_id": pdf_id,
                "page": num_pagina,
                "text": texto
            }

            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"✓ Texto extraído: {pdf_id}")

if __name__ == "__main__":
    carpeta_pdfs = "data/pdfs"
    carpeta_salida = "data/extracted"

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    for pdf in Path(carpeta_pdfs).glob("*.pdf"):
        extraer_texto_pdf(pdf, carpeta_salida)

if __name__ == "__main__":
    carpeta_pdfs = "data/pdfs"
    carpeta_salida = "data/extracted"

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    for pdf in Path(carpeta_pdfs).glob("*.pdf"):
        extraer_texto_pdf(pdf, carpeta_salida)
