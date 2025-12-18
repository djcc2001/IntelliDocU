import json
import re
from pathlib import Path

SECTION_PATTERNS = [
    (r"^\s*resumen\s*$", "abstract"),
    (r"^\s*abstract\s*$", "abstract"),
    (r"^\s*introducci[óo]n\s*$", "introduction"),
    (r"^\s*introduction\s*$", "introduction"),
    (r"^\s*(materiales?\s+y\s+)?m[ée]todos?\s*$", "method"),
    (r"^\s*(materials?\s+and\s+)?methods?\s*$", "method"),
    (r"^\s*metodolog[íi]a\s*$", "method"),
    (r"^\s*methodology\s*$", "method"),
    (r"^\s*resultados?\s*$", "results"),
    (r"^\s*results?\s*$", "results"),
    (r"^\s*discusi[óo]n\s*$", "discussion"),
    (r"^\s*discussion\s*$", "discussion"),
    (r"^\s*conclusi[óo]n(es)?\s*$", "conclusion"),
    (r"^\s*conclusions?\s*$", "conclusion"),
    (r"^\s*(referencias?|bibliography)\s*$", "references"),
    (r"^\s*(ap[ée]ndice|appendi(x|ces))\s*$", "appendix"),
]

def detectar_seccion(texto):
    lineas = texto.split("\n")[:10]

    for linea in lineas:
        linea = linea.strip().lower()

        if len(linea) == 0 or len(linea) > 40:
            continue

        if re.match(r"^\d+$", linea):
            continue

        for patron, seccion in SECTION_PATTERNS:
            if re.match(patron, linea, re.IGNORECASE):
                return seccion

    return None

def limpiar_texto(texto):
    # Unir palabras cortadas por guión
    texto = re.sub(r"-\s*\n\s*", "", texto)

    # Preservar párrafos
    texto = re.sub(r"\n{2,}", "\n\n", texto)
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)

    # Espacios extra
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

def limpiar_archivo(ruta_entrada, carpeta_salida):
    pdf_id = Path(ruta_entrada).stem
    ruta_salida = Path(carpeta_salida) / f"{pdf_id}.jsonl"

    seccion_actual = "unknown"
    paginas = 0

    with open(ruta_entrada, "r", encoding="utf-8") as fin, \
         open(ruta_salida, "w", encoding="utf-8") as fout:

        for linea in fin:
            registro = json.loads(linea)
            texto_original = registro["text"]

            # Detectar sección solo si es probable encabezado
            if len(texto_original) < 500:
                nueva = detectar_seccion(texto_original)
                if nueva:
                    seccion_actual = nueva

            texto_limpio = limpiar_texto(texto_original)

            if len(texto_limpio) < 80:
                continue

            registro["clean_text"] = texto_limpio
            registro["section"] = seccion_actual

            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
            paginas += 1

    print(f"✓ {pdf_id}: {paginas} páginas limpias")

if __name__ == "__main__":
    entrada = Path("data/extracted")
    salida = Path("data/preprocessed")

    salida.mkdir(parents=True, exist_ok=True)

    archivos = list(entrada.glob("*.jsonl"))
    print(f"Procesando {len(archivos)} archivo(s)...\n")

    for archivo in archivos:
        limpiar_archivo(archivo, salida)

    print("\n✓ Limpieza completada")
