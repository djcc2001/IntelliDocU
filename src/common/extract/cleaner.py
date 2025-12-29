"""
Modulo de limpieza y normalizacion de texto extraido de PDFs.
Detecta secciones del documento y limpia el texto eliminando saltos de linea
innecesarios y espacios en exceso.
"""

import json
import re
from pathlib import Path


# Patrones para detectar secciones comunes en documentos academicos
PATRONES_SECCION = [
    (r"^\s*resumen\s*$", "abstract"),
    (r"^\s*abstract\s*$", "abstract"),
    (r"^\s*introduccion\s*$", "introduction"),
    (r"^\s*introduction\s*$", "introduction"),
    (r"^\s*(materiales?\s+y\s+)?metodos?\s*$", "method"),
    (r"^\s*(materials?\s+and\s+)?methods?\s*$", "method"),
    (r"^\s*metodologia\s*$", "method"),
    (r"^\s*methodology\s*$", "method"),
    (r"^\s*resultados?\s*$", "results"),
    (r"^\s*results?\s*$", "results"),
    (r"^\s*discusion\s*$", "discussion"),
    (r"^\s*discussion\s*$", "discussion"),
    (r"^\s*conclusion(es)?\s*$", "conclusion"),
    (r"^\s*conclusions?\s*$", "conclusion"),
    (r"^\s*(referencias?|bibliography)\s*$", "references"),
    (r"^\s*(apendice|appendi(x|ces))\s*$", "appendix"),
]


def detectar_seccion(texto):
    """
    Detecta la seccion del documento basandose en las primeras lineas del texto.
    
    Args:
        texto: Texto de la pagina a analizar
    
    Returns:
        Nombre de la seccion detectada o None si no se encuentra ninguna
    """
    lineas = texto.split("\n")[:10]  # Analizar solo las primeras 10 lineas

    for linea in lineas:
        linea = linea.strip().lower()

        # Ignorar lineas vacias o muy largas (probablemente no son encabezados)
        if len(linea) == 0 or len(linea) > 40:
            continue

        # Ignorar lineas que son solo numeros
        if re.match(r"^\d+$", linea):
            continue

        # Buscar coincidencias con los patrones de seccion
        for patron, seccion in PATRONES_SECCION:
            if re.match(patron, linea, re.IGNORECASE):
                return seccion

    return None


def limpiar_texto(texto):
    """
    Limpia y normaliza el texto eliminando:
    - Guiones al final de linea (palabras cortadas)
    - Saltos de linea multiples
    - Espacios en exceso
    
    Args:
        texto: Texto original a limpiar
    
    Returns:
        Texto limpio y normalizado
    """
    # Unir palabras cortadas por guion al final de linea
    texto = re.sub(r"-\s*\n\s*", "", texto)

    # Preservar parrafos (dobles saltos de linea)
    texto = re.sub(r"\n{2,}", "\n\n", texto)
    # Convertir saltos de linea simples en espacios
    texto = re.sub(r"(?<!\n)\n(?!\n)", " ", texto)

    # Eliminar espacios multiples
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def limpiar_archivo(ruta_entrada, carpeta_salida):
    """
    Procesa un archivo JSONL de texto extraido, limpia el texto y detecta secciones.
    
    Args:
        ruta_entrada: Ruta al archivo JSONL con texto extraido
        carpeta_salida: Directorio donde se guardara el archivo JSONL limpio
    
    Genera un archivo JSONL con los campos originales mas:
    - clean_text: Texto limpio y normalizado
    - section: Seccion detectada del documento
    """
    identificador_pdf = Path(ruta_entrada).stem
    ruta_salida = Path(carpeta_salida) / f"{identificador_pdf}.jsonl"

    seccion_actual = "unknown"
    paginas_procesadas = 0

    with open(ruta_entrada, "r", encoding="utf-8") as archivo_entrada, \
         open(ruta_salida, "w", encoding="utf-8") as archivo_salida:

        for linea in archivo_entrada:
            registro = json.loads(linea)
            texto_original = registro["text"]

            # Detectar seccion solo si el texto es corto (probable encabezado)
            if len(texto_original) < 500:
                nueva_seccion = detectar_seccion(texto_original)
                if nueva_seccion:
                    seccion_actual = nueva_seccion

            texto_limpio = limpiar_texto(texto_original)

            # Ignorar paginas con muy poco texto (probablemente imagenes o vacias)
            if len(texto_limpio) < 80:
                continue

            registro["clean_text"] = texto_limpio
            registro["section"] = seccion_actual

            archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            paginas_procesadas += 1

    print(f"✓ {identificador_pdf}: {paginas_procesadas} paginas limpias")


if __name__ == "__main__":
    # Configuracion de rutas por defecto
    carpeta_entrada = Path("data/extracted")
    carpeta_salida = Path("data/preprocessed")

    # Crear directorio de salida si no existe
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Procesar todos los archivos JSONL en la carpeta de entrada
    archivos = list(carpeta_entrada.glob("*.jsonl"))
    print(f"Procesando {len(archivos)} archivo(s)...\n")

    for archivo in archivos:
        limpiar_archivo(archivo, carpeta_salida)

    print("\n✓ Limpieza completada")
