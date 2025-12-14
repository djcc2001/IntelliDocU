import json
import re
from pathlib import Path

# Jerarquía: appendix primero, luego abstract, introduction, etc.
SECTION_PATTERNS = [
    (r"\bappendix\b|\ba\.1\b|\ba\.\b", "appendix"),
    (r"\babstract\b", "abstract"),
    (r"\bintroduction\b", "introduction"),
    (r"\brelated work\b", "related_work"),
    (r"\bmethod\b|\bmethodology\b|\bapproach\b", "method"),
    (r"\bexperiments?\b|\bevaluation\b|\bstudy\b", "experiments"),
    (r"\bresults?\b", "results"),
    (r"\bdiscussion\b", "discussion"),
    (r"\bconclusion\b|\bconclusions\b", "conclusion"),
]

def detectar_seccion(texto):
    """
    Detecta la sección de un fragmento de texto.
    Primero appendix, luego abstract, introduction, etc.
    """
    texto_lower = texto.lower()[:1000]  # mirar los primeros 1000 caracteres
    for patron, nombre in SECTION_PATTERNS:
        if re.search(patron, texto_lower):
            return nombre
    return None



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

    seccion_actual = "unknown"

    with open(ruta_entrada, "r", encoding="utf-8") as fin, \
         open(ruta_salida, "w", encoding="utf-8") as fout:

        for linea in fin:
            registro = json.loads(linea)
            texto_limpio = limpiar_texto(registro["text"])

            nueva_seccion = detectar_seccion(texto_limpio)
            if nueva_seccion:
                seccion_actual = nueva_seccion

            registro["clean_text"] = texto_limpio
            registro["section"] = seccion_actual

            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"Texto limpio generado: {pdf_id}")

if __name__ == "__main__":
    carpeta_entrada = "data/extracted"
    carpeta_salida = "data/preprocessed"

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    for archivo in Path(carpeta_entrada).glob("*.jsonl"):
        limpiar_archivo(archivo, carpeta_salida)
