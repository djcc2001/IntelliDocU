import json
import faiss
import numpy as np
from embedder import Embedder

# ===============================
# Configuración
# ===============================
INDEX_DIR = "data/indices/faiss"
QUERY = "SparseSwaps pruning mask refinement algorithm"
K = 5  # Número de resultados a devolver

# ===============================
# Cargar índice FAISS
# ===============================
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")

# Cargar mapping
with open(f"{INDEX_DIR}/mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)

# ===============================
# Inicializar embedder
# ===============================
embedder = Embedder()

# ===============================
# Generar embedding de la consulta
# ===============================
query_embedding = embedder.encode([QUERY])[0].astype("float32")  # embedding normalizado

# ===============================
# Búsqueda en FAISS
# ===============================
distances, indices = index.search(query_embedding.reshape(1, -1), K)

# ===============================
# Mostrar resultados
# ===============================
print(f"\nResultados para la consulta: '{QUERY}'\n")

for rank, idx in enumerate(indices[0]):
    result = mapping[idx]
    print(f"{rank + 1}. Doc: {result['doc_id']}")
    print(f"   Page: {result['page']}")
    print(f"   Section: {result.get('section')}")
    print(f"   FragID: {result['frag_id']}")
    print(f"   Similitud: {distances[0][rank]:.4f}")
    print(f"   Texto: {result['text'][:200]}...\n")
