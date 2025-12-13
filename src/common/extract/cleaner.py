import json
import re
from pathlib import Path

def limpiar_texto(texto):
    # Unir palabras cortadas por guion al final de linea
    texto = re.sub(r"-\n", "", texto)

    # Reemplazar saltos de linea por espacio
    texto = texto.replace("\n", " ")

    # Eliminar espacios duplicados
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

def limpiar_archivo(ruta_entrada, carpeta_salida):
    pdf_id = Path(ruta_entrada).stem
    ruta_salida = Path(carpeta_salida) / f"{pdf_id}.jsonl"

    with open(ruta_entrada, "r", encoding="utf-8") as fin, \
         open(ruta_salida, "w", encoding="utf-8") as fout:

        for linea in fin:
            registro = json.loads(linea)
            texto_limpio = limpiar_texto(registro["text"])

            registro["clean_text"] = texto_limpio
            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"Texto limpio generado: {pdf_id}")

if __name__ == "__main__":
    carpeta_entrada = "data/extracted"
    carpeta_salida = "data/preprocessed"

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    for archivo in Path(carpeta_entrada).glob("*.jsonl"):
        limpiar_archivo(archivo, carpeta_salida)
