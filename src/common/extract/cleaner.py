import json
import re
from pathlib import Path

# Patrones de sección en orden de prioridad (más específicos primero)
# Busca patrones que indiquen TÍTULOS de sección, no menciones en el texto
SECTION_PATTERNS = [
    # Español
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
    (r"^\s*referencias?\s*(bibliogr[áa]ficas?)?\s*$", "references"),
    (r"^\s*(references|bibliography)\s*$", "references"),
    (r"^\s*(ap[ée]ndice|anexo)[sa]?\s*$", "appendix"),
    (r"^\s*appendi(x|ces)\s*$", "appendix"),
]

def detectar_seccion(texto):
    """
    Detecta la sección buscando títulos en las primeras líneas.
    Solo considera líneas cortas que parezcan títulos.
    """
    lineas = texto.split('\n')[:15]  # Primeras 15 líneas
    
    for linea in lineas:
        linea_limpia = linea.strip().lower()
        
        # Ignorar líneas muy largas (probablemente no son títulos)
        if len(linea_limpia) > 50:
            continue
            
        # Ignorar líneas con números de página o metadata
        if re.match(r'^\d+\s*$', linea_limpia):
            continue
            
        # Buscar coincidencias con patrones de sección
        for patron, nombre_seccion in SECTION_PATTERNS:
            if re.match(patron, linea_limpia, re.IGNORECASE):
                return nombre_seccion
    
    return None

def limpiar_texto(texto):
    """Limpia el texto eliminando artefactos de PDF"""
    # Unir palabras cortadas por guion al final de línea
    texto = re.sub(r"-\n", "", texto)
    
    # Reemplazar saltos de línea por espacio
    texto = texto.replace("\n", " ")
    
    # Eliminar espacios múltiples
    texto = re.sub(r"\s+", " ", texto)
    
    return texto.strip()

def limpiar_archivo(ruta_entrada, carpeta_salida):
    """Procesa un archivo JSONL y genera versión limpia"""
    pdf_id = Path(ruta_entrada).stem
    ruta_salida = Path(carpeta_salida) / f"{pdf_id}.jsonl"

    seccion_actual = "unknown"
    paginas_procesadas = 0

    with open(ruta_entrada, "r", encoding="utf-8") as fin, \
         open(ruta_salida, "w", encoding="utf-8") as fout:

        for linea in fin:
            registro = json.loads(linea)
            texto_original = registro["text"]
            texto_limpio = limpiar_texto(texto_original)

            # Detectar sección usando texto ORIGINAL (mantiene saltos de línea)
            nueva_seccion = detectar_seccion(texto_original)
            if nueva_seccion:
                seccion_actual = nueva_seccion

            registro["clean_text"] = texto_limpio
            registro["section"] = seccion_actual

            fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
            paginas_procesadas += 1

    print(f"✓ {pdf_id}: {paginas_procesadas} páginas procesadas")

if __name__ == "__main__":
    carpeta_entrada = "data/extracted"
    carpeta_salida = "data/preprocessed"

    Path(carpeta_salida).mkdir(parents=True, exist_ok=True)

    archivos = list(Path(carpeta_entrada).glob("*.jsonl"))
    print(f"Procesando {len(archivos)} archivo(s)...\n")
    
    for archivo in archivos:
        limpiar_archivo(archivo, carpeta_salida)
    
    print(f"\n✓ Limpieza completada: {len(archivos)} archivo(s)")