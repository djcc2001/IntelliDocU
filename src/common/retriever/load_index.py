import faiss
import json
from pathlib import Path


def load_faiss_index(base_data_dir="data"):
    base_data_dir = Path(base_data_dir)
    index_path = base_data_dir / "indices" / "faiss" / "index.faiss"
    return faiss.read_index(str(index_path))


def load_mapping(base_data_dir="data"):
    base_data_dir = Path(base_data_dir)
    mapping_path = base_data_dir / "indices" / "faiss" / "mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)
