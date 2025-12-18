import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.common.embeddings.embedder import Embedder

BATCH_SIZE = 32

def build_faiss_index(base_data_dir="data"):
    base = Path(base_data_dir)

    fragments_dir = base / "fragments"
    index_dir = base / "indices" / "faiss"
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder()

    all_embeddings = []
    mapping = []

    fragment_files = list(fragments_dir.glob("*.jsonl"))
    if not fragment_files:
        raise RuntimeError("❌ No se encontraron fragments")

    for frag_file in tqdm(fragment_files, desc="FAISS indexing"):
        batch_texts = []
        batch_meta = []

        with open(frag_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                # Filtrar referencias
                if data.get("section") == "references":
                    continue

                text = data.get("text", "").strip()
                if not text:
                    continue

                batch_texts.append(text)
                batch_meta.append({
                    "doc_id": data["doc_id"],
                    "section": data.get("section"),
                    "pages": data.get("pages"),
                    "frag_id": data.get("frag_id"),
                    "chunk_in_section": data.get("chunk_in_section")
                })

                if len(batch_texts) == BATCH_SIZE:
                    _process_batch(embedder, batch_texts, batch_meta, all_embeddings, mapping)
                    batch_texts, batch_meta = [], []

        if batch_texts:
            _process_batch(embedder, batch_texts, batch_meta, all_embeddings, mapping)

    if not all_embeddings:
        raise RuntimeError("❌ No se generaron embeddings")

    embeddings = np.vstack(all_embeddings).astype("float32")

    # Validación de dimensión
    assert embeddings.shape[1] == embedder.dim, "Dimensión incorrecta de embeddings"

    index = faiss.IndexFlatIP(embedder.dim)
    index.add(embeddings)

    # Guardar índice
    faiss.write_index(index, str(index_dir / "index.faiss"))

    # Guardar mapping (ligero)
    with open(index_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    # Guardar metadata del índice
    with open(index_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "embedding_model": embedder.model_name,
            "dimension": embedder.dim,
            "normalized": True,
            "similarity": "cosine",
            "num_vectors": index.ntotal
        }, f, indent=2)

    print(f"✓ FAISS listo — vectores: {index.ntotal}")


def _process_batch(embedder, texts, metas, all_embeddings, mapping):
    embeddings = embedder.encode(texts)

    for emb, meta in zip(embeddings, metas):
        all_embeddings.append(emb)
        mapping.append(meta)


def main():
    try:
        build_faiss_index(base_data_dir="data")
    except Exception as e:
        print(f"[ERROR] FAISS falló: {e}")


if __name__ == "__main__":
    main()
