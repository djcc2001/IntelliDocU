"""
Modulo para construir el indice vectorial FAISS a partir de fragmentos de texto.
FAISS permite busqueda rapida de similitud semantica usando embeddings.
"""

import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.common.embeddings.embedder import GeneradorEmbeddings

TAMANO_LOTE = 32  # Numero de textos a procesar por lote


def construir_indice_faiss(directorio_base_datos="data"):
    """
    Construye un indice FAISS a partir de todos los fragmentos de texto.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Genera:
    - indices/faiss/index.faiss: Indice FAISS para busqueda rapida
    - indices/faiss/mapping.json: Mapeo de indices a metadatos de fragmentos
    - indices/faiss/index_meta.json: Metadatos del indice (modelo, dimension, etc.)
    """
    base = Path(directorio_base_datos)

    directorio_fragmentos = base / "fragments"
    directorio_indices = base / "indices" / "faiss"
    directorio_indices.mkdir(parents=True, exist_ok=True)

    generador_embeddings = GeneradorEmbeddings()

    todos_los_embeddings = []
    mapeo = []

    archivos_fragmentos = list(directorio_fragmentos.glob("*.jsonl"))
    if not archivos_fragmentos:
        raise RuntimeError("NO se encontraron fragmentos")

    for archivo_fragmento in tqdm(archivos_fragmentos, desc="Indexando FAISS"):
        textos_lote = []
        metadatos_lote = []

        with open(archivo_fragmento, "r", encoding="utf-8") as archivo:
            for linea in archivo:
                datos = json.loads(linea)

                # Filtrar referencias bibliograficas (no utiles para busqueda)
                if datos.get("section") == "references":
                    continue

                texto = datos.get("text", "").strip()
                if not texto:
                    continue

                textos_lote.append(texto)
                
                # Asegurar que pages siempre sea una lista valida
                paginas = datos.get("pages")
                if paginas is None:
                    # Fallback: buscar 'page' (singular) si 'pages' no existe
                    pagina_singular = datos.get("page")
                    if pagina_singular is not None:
                        paginas = [pagina_singular]
                    else:
                        paginas = []
                
                # Asegurar que sea lista
                if not isinstance(paginas, list):
                    paginas = [paginas] if paginas is not None else []
                
                metadatos_lote.append({
                    "doc_id": datos["doc_id"],
                    "section": datos.get("section"),
                    "pages": paginas,  # Siempre lista
                    "frag_id": datos.get("frag_id"),
                    "chunk_in_section": datos.get("chunk_in_section"),
                    "text": texto 
                })

                # Procesar cuando el lote esta lleno
                if len(textos_lote) == TAMANO_LOTE:
                    _procesar_lote(generador_embeddings, textos_lote, metadatos_lote, todos_los_embeddings, mapeo)
                    textos_lote, metadatos_lote = [], []

        # Procesar lote restante
        if textos_lote:
            _procesar_lote(generador_embeddings, textos_lote, metadatos_lote, todos_los_embeddings, mapeo)

    if not todos_los_embeddings:
        raise RuntimeError("NO se generaron embeddings")

    # Convertir lista de arrays a una matriz numpy
    embeddings = np.vstack(todos_los_embeddings).astype("float32")

    # Validar dimension
    assert embeddings.shape[1] == generador_embeddings.dimension, "Dimension incorrecta de embeddings"

    # Crear indice FAISS con producto interno (para embeddings normalizados = similitud coseno)
    indice = faiss.IndexFlatIP(generador_embeddings.dimension)
    indice.add(embeddings)

    # Guardar indice
    faiss.write_index(indice, str(directorio_indices / "index.faiss"))

    # Guardar mapeo (ligero, solo metadatos)
    with open(directorio_indices / "mapping.json", "w", encoding="utf-8") as archivo:
        json.dump(mapeo, archivo, ensure_ascii=False, indent=2)

    # Guardar metadatos del indice
    with open(directorio_indices / "index_meta.json", "w", encoding="utf-8") as archivo:
        json.dump({
            "embedding_model": generador_embeddings.nombre_modelo,
            "dimension": generador_embeddings.dimension,
            "normalized": True,
            "similarity": "cosine",
            "num_vectors": indice.ntotal
        }, archivo, indent=2)

    print(f"✓ FAISS listo — vectores: {indice.ntotal}")


def _procesar_lote(generador_embeddings, textos, metadatos, todos_los_embeddings, mapeo):
    """
    Procesa un lote de textos generando embeddings y agregandolos a las listas.
    
    Args:
        generador_embeddings: Instancia de GeneradorEmbeddings
        textos: Lista de textos a procesar
        metadatos: Lista de metadatos correspondientes a cada texto
        todos_los_embeddings: Lista donde se agregaran los embeddings
        mapeo: Lista donde se agregaran los metadatos
    """
    embeddings = generador_embeddings.codificar(textos)

    for embedding, metadato in zip(embeddings, metadatos):
        todos_los_embeddings.append(embedding)
        mapeo.append(metadato)


def main():
    """Funcion principal para ejecutar la construccion del indice."""
    try:
        construir_indice_faiss(directorio_base_datos="data")
    except Exception as error:
        print(f"[ERROR] FAISS fallo: {error}")


if __name__ == "__main__":
    main()
