"""
Script de validacion del indice FAISS.
Permite probar la busqueda en el indice con consultas de ejemplo.
"""

import json
import faiss
import numpy as np
from src.common.embeddings.embedder import GeneradorEmbeddings

# ===============================
# Configuracion
# ===============================
DIRECTORIO_INDICES = "data/indices/faiss"
CONSULTA = "SparseSwaps pruning mask refinement algorithm"
K = 5  # Numero de resultados a devolver

# ===============================
# Cargar indice FAISS
# ===============================
indice = faiss.read_index(f"{DIRECTORIO_INDICES}/index.faiss")

# Cargar mapeo
with open(f"{DIRECTORIO_INDICES}/mapping.json", "r", encoding="utf-8") as archivo:
    mapeo = json.load(archivo)

# ===============================
# Inicializar generador de embeddings
# ===============================
generador_embeddings = GeneradorEmbeddings()

# ===============================
# Generar embedding de la consulta
# ===============================
embedding_consulta = generador_embeddings.codificar([CONSULTA])[0].astype("float32")  # embedding normalizado

# ===============================
# Busqueda en FAISS
# ===============================
distancias, indices_resultados = indice.search(embedding_consulta.reshape(1, -1), K)

# ===============================
# Mostrar resultados
# ===============================
print(f"\nResultados para la consulta: '{CONSULTA}'\n")

for rango, indice_resultado in enumerate(indices_resultados[0]):
    resultado = mapeo[indice_resultado]
    print(f"{rango + 1}. Doc: {resultado['doc_id']}")
    print(f"   Pagina: {resultado.get('pages', resultado.get('page', '?'))}")
    print(f"   Seccion: {resultado.get('section')}")
    print(f"   FragID: {resultado['frag_id']}")
    print(f"   Similitud: {distancias[0][rango]:.4f}")
    print(f"   Texto: {resultado['text'][:200]}...\n")
