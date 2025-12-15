import json
import faiss
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from src.common.embeddings.embedder import Embedder

BATCH_SIZE = 32

def build_faiss_index(base_data_dir="data"):
    base_data_dir = Path(base_data_dir)

    fragments_dir = base_data_dir / "fragments"
    index_dir = base_data_dir / "indices" / "faiss"
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = Embedder()

    all_embeddings = []
    mapping = []
    vector_id = 0

    fragment_files = list(fragments_dir.glob("*.jsonl"))
    if not fragment_files:
        raise RuntimeError("No se encontraron fragments")

    for frag_file in tqdm(fragment_files, desc="FAISS indexing"):
        batch_texts = []
        batch_data = []

        with open(frag_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)

                if data.get("section") == "references":
                    continue

                text = data["text"].strip()
                if not text:
                    continue

                batch_texts.append(text)
                batch_data.append(data)

                if len(batch_texts) == BATCH_SIZE:
                    _process_batch(
                        embedder,
                        batch_texts,
                        batch_data,
                        all_embeddings,
                        mapping,
                        vector_id
                    )
                    vector_id += len(batch_texts)
                    batch_texts, batch_data = [], []

        if batch_texts:
            _process_batch(
                embedder,
                batch_texts,
                batch_data,
                all_embeddings,
                mapping,
                vector_id
            )
            vector_id += len(batch_texts)

    if not all_embeddings:
        raise RuntimeError("No se generaron embeddings")

    embeddings_matrix = np.vstack(all_embeddings).astype("float32")

    dim = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_matrix)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    with open(index_dir / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"FAISS listo — vectores: {index.ntotal}")


def _process_batch(embedder, texts, datas, all_embeddings, mapping, start_id):
    embeddings = embedder.encode(texts)
    for i, (emb, dat) in enumerate(zip(embeddings, datas)):
        all_embeddings.append(emb)
        mapping.append({
            "vector_id": start_id + i,
            "doc_id": dat["doc_id"],
            "page": dat["page"],
            "section": dat.get("section"),
            "frag_id": dat["frag_id"],
            "text": dat["text"]
        })

def main():
    try:
        build_faiss_index(base_data_dir="data")
    except Exception as e:
        print(f"[ERROR] No se pudo construir FAISS: {e}")


if __name__ == "__main__":
    main()
