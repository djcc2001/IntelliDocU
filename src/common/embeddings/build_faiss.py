import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from embedder import Embedder

# ===============================
# Configuración de rutas
# ===============================
FRAGMENTS_DIR = Path("data/fragments")
INDEX_DIR = Path("indices/faiss_global")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# Parámetros de batch
# ===============================
BATCH_SIZE = 32  # Número de fragments a procesar por batch

# ===============================
# Construcción del índice FAISS
# ===============================
def build_index():
    embedder = Embedder()

    all_embeddings = []
    mapping = []
    vector_id = 0

    fragment_files = list(FRAGMENTS_DIR.glob("*.jsonl"))

    if not fragment_files:
        print("No se encontraron fragments en", FRAGMENTS_DIR)
        return

    for frag_file in tqdm(fragment_files, desc="Procesando fragments"):
        batch_texts = []
        batch_data = []

        with open(frag_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Excluir referencias
                if data.get("section") == "references":
                    continue

                text = data["text"].strip()
                if not text:
                    continue

                batch_texts.append(text)
                batch_data.append(data)

                # Procesar batch completo
                if len(batch_texts) == BATCH_SIZE:
                    embeddings = embedder.encode(batch_texts)
                    for emb, dat in zip(embeddings, batch_data):
                        all_embeddings.append(emb)
                        mapping.append({
                            "vector_id": vector_id,
                            "doc_id": dat["doc_id"],
                            "page": dat["page"],
                            "section": dat.get("section"),
                            "frag_id": dat["frag_id"],
                            "text": dat["text"]
                        })
                        vector_id += 1
                    batch_texts, batch_data = [], []

        # Procesar batch final si queda algo
        if batch_texts:
            embeddings = embedder.encode(batch_texts)
            for emb, dat in zip(embeddings, batch_data):
                all_embeddings.append(emb)
                mapping.append({
                    "vector_id": vector_id,
                    "doc_id": dat["doc_id"],
                    "page": dat["page"],
                    "section": dat.get("section"),
                    "frag_id": dat["frag_id"],
                    "text": dat["text"]
                })
                vector_id += 1

    if not all_embeddings:
        raise RuntimeError("No se generaron embeddings. Revisa los fragments.")

    # Convertir a matriz numpy
    embeddings_matrix = np.vstack(all_embeddings).astype("float32")

    # Crear índice FAISS (cosine similarity usando inner product con embeddings normalizados)
    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    # Guardar índice y mapping
    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print("Índice FAISS construido correctamente")
    print(f"Total vectores: {index.ntotal}")


if __name__ == "__main__":
    build_index()
