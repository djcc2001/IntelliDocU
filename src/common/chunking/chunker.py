"""
Modulo de fragmentacion (chunking) de texto para RAG.
Divide el texto en fragmentos de tamano optimo basado en tokens del modelo de embeddings,
manteniendo continuidad semantica agrupando por secciones.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


# ===============================
# Configuracion RAG-aware
# ===============================
MODELO_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
TAMANO_CHUNK = 384  # Tamano de fragmento en tokens
SOLAPAMIENTO = 64   # Tokens de solapamiento entre fragmentos
MIN_TOKENS_CHUNK = 50  # Minimo de tokens para considerar un fragmento valido

print("Cargando tokenizer de embeddings...")
tokenizer = AutoTokenizer.from_pretrained(MODELO_EMBEDDING)


# -------------------------------------------------
# Fragmentacion basada en tokens de EMBEDDINGS
# -------------------------------------------------
def fragmentar_tokens(tokens, tamano_chunk, solapamiento):
    """
    Divide una lista de tokens en fragmentos con solapamiento.
    
    Args:
        tokens: Lista de tokens a fragmentar
        tamano_chunk: Tamano maximo de cada fragmento en tokens
        solapamiento: Numero de tokens de solapamiento entre fragmentos
    
    Returns:
        Lista de tuplas (inicio, fin, tokens_fragmento)
    """
    inicio = 0
    fragmentos = []

    while inicio < len(tokens):
        fin = min(inicio + tamano_chunk, len(tokens))
        fragmento = tokens[inicio:fin]

        # Si el fragmento es muy pequeno y no es el primero, terminar
        if len(fragmento) < MIN_TOKENS_CHUNK and inicio > 0:
            break

        fragmentos.append((inicio, fin, fragmento))

        # Si llegamos al final, terminar
        if fin == len(tokens):
            break

        # Avanzar con solapamiento
        inicio = fin - solapamiento

    return fragmentos


def procesar_archivo(ruta_jsonl, directorio_salida):
    """
    Procesa un archivo JSONL preprocesado y genera fragmentos para RAG.
    
    Args:
        ruta_jsonl: Ruta al archivo JSONL con texto preprocesado
        directorio_salida: Directorio donde se guardara el archivo de fragmentos
    
    Genera un archivo JSONL con fragmentos conteniendo:
    - doc_id: Identificador del documento
    - section: Seccion del documento
    - pages: Lista de paginas que contiene el fragmento
    - frag_id: Identificador unico del fragmento
    - chunk_in_section: Indice del fragmento dentro de su seccion
    - text: Texto del fragmento
    - token_count: Numero de tokens del fragmento
    """
    identificador_doc = ruta_jsonl.stem
    ruta_salida = directorio_salida / f"{identificador_doc}_fragments.jsonl"

    identificador_fragmento = 0
    fragmentos_generados = 0

    with open(ruta_jsonl, "r", encoding="utf-8") as archivo_entrada, \
         open(ruta_salida, "w", encoding="utf-8") as archivo_salida:

        # Agrupar por seccion para mantener continuidad semantica
        secciones = {}
        for linea in archivo_entrada:
            datos = json.loads(linea)
            seccion = datos.get("section", "unknown")
            secciones.setdefault(seccion, []).append(datos)

        for seccion, items in secciones.items():
            # Preparar lista de textos por pagina
            textos_por_pagina = []
            for item in items:
                texto = item.get("clean_text")
                pagina = item.get("page")
                if texto:
                    textos_por_pagina.append((pagina, texto))

            if not textos_por_pagina:
                continue

            # Unir todo el texto de la seccion
            texto_total = "\n\n".join(texto for _, texto in textos_por_pagina)
            tokens = tokenizer(texto_total, add_special_tokens=False, truncation=False)["input_ids"]

            # Mapear indices de tokens a paginas para saber de que pagina viene cada token
            mapeo_token_a_pagina = []
            for pagina, texto in textos_por_pagina:
                tokens_texto = tokenizer(texto, add_special_tokens=False)["input_ids"]
                mapeo_token_a_pagina.extend([pagina] * len(tokens_texto))

            # Fragmentar tokens
            fragmentos_tokens = fragmentar_tokens(tokens, TAMANO_CHUNK, SOLAPAMIENTO)

            for indice, (inicio, fin, ids_fragmento) in enumerate(fragmentos_tokens):
                texto_fragmento = tokenizer.decode(ids_fragmento, skip_special_tokens=True)
                paginas_en_fragmento = sorted(set(mapeo_token_a_pagina[inicio:fin]))

                registro = {
                    "doc_id": identificador_doc,
                    "section": seccion,
                    "pages": paginas_en_fragmento,
                    "frag_id": identificador_fragmento,
                    "chunk_in_section": indice,
                    "text": texto_fragmento,
                    "token_count": len(ids_fragmento)
                }

                archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
                identificador_fragmento += 1
                fragmentos_generados += 1

    paginas_unicas = len(set(pagina for pagina, _ in textos_por_pagina)) if 'textos_por_pagina' in locals() else 0
    print(f"  ✓ {identificador_doc}: {paginas_unicas} paginas → {fragmentos_generados} fragmentos")


def generar_chunks(directorio_base_datos="data"):
    """
    Genera fragmentos para todos los archivos preprocesados en el directorio.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos (debe contener carpeta 'preprocessed')
    """
    base = Path(directorio_base_datos)
    directorio_entrada = base / "preprocessed"
    directorio_salida = base / "fragments"

    directorio_salida.mkdir(parents=True, exist_ok=True)

    archivos = list(directorio_entrada.glob("*.jsonl"))
    if not archivos:
        print("❌ No hay archivos preprocessed")
        return

    print("\nIniciando Fase 3: Fragmentacion (RAG-aware)")
    print(f"  Embeddings: {MODELO_EMBEDDING}")
    print(f"  Tamano fragmento: {TAMANO_CHUNK}")
    print(f"  Solapamiento: {SOLAPAMIENTO}\n")

    for archivo in tqdm(archivos, desc="Fragmentando"):
        procesar_archivo(archivo, directorio_salida)

    # Contar total de fragmentos generados
    total = sum(
        1 for archivo_fragmentos in directorio_salida.glob("*_fragments.jsonl")
        for _ in open(archivo_fragmentos, encoding="utf-8")
    )

    print("\n✓ Fase 3 completada")
    print(f"  Total fragmentos: {total}")
    print(f"  Salida: {directorio_salida}")


if __name__ == "__main__":
    generar_chunks()
