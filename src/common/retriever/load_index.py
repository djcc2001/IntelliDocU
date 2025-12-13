# Carga del indice FAISS y del mapeo de fragmentos

import faiss
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]

INDEX_PATH = BASE_DIR / "indices" / "faiss_global" / "index.faiss"
MAPPING_PATH = BASE_DIR / "indices" / "faiss_global" / "mapping.json"


def load_faiss_index():
    index = faiss.read_index(str(INDEX_PATH))
    return index


def load_mapping():
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping
