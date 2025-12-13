import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from embedder import Embedder


FRAGMENTS_DIR = Path("data/fragments")
INDEX_DIR = Path("indices/faiss_global")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_index():
    embedder = Embedder()

    all_embeddings = []
    mapping = []
    vector_id = 0

    fragment_files = list(FRAGMENTS_DIR.glob("*.jsonl"))

    for frag_file in tqdm(fragment_files, desc="Procesando fragmentos"):
        with open(frag_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                text = data["text"]
                embedding = embedder.encode([text])[0]

                all_embeddings.append(embedding)

                mapping.append({
                    "vector_id": vector_id,
                    "doc_id": data["doc_id"],
                    "page": data["page"],
                    "frag_id": data["frag_id"],
                    "text": text
                })

                vector_id += 1

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")

    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

    with open(INDEX_DIR / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print("Indice FAISS construido correctamente")
    print(f"Total vectores: {index.ntotal}")


if __name__ == "__main__":
    build_index()
