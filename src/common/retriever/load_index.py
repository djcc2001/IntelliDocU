import faiss
import json
from pathlib import Path


def load_faiss_index(base_data_dir="data"):
    base = Path(base_data_dir)
    index_path = base / "indices" / "faiss" / "index.faiss"
    return faiss.read_index(str(index_path))


def load_mapping(base_data_dir="data"):
    base = Path(base_data_dir)
    mapping_path = base / "indices" / "faiss" / "mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index_meta(base_data_dir="data"):
    base = Path(base_data_dir)
    meta_path = base / "indices" / "faiss" / "index_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}
