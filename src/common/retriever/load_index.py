import faiss
import json
from pathlib import Path

def load_faiss_index(base_data_dir="data"):
    base = Path(base_data_dir)
    index_path = base / "indices" / "faiss" / "index.faiss"

    # 🚫 BLOQUEO TOTAL
    if not index_path.exists():
        return None

    try:
        if index_path.stat().st_size == 0:
            return None
    except Exception:
        return None

    # ⛔ ESTA LÍNEA SOLO SE EJECUTA SI EL ARCHIVO ES REAL
    return faiss.read_index(str(index_path))



def load_mapping(base_data_dir="data"):
    base = Path(base_data_dir)
    mapping_path = base / "indices" / "faiss" / "mapping.json"

    if not mapping_path.exists():
        return []

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def load_index_meta(base_data_dir="data"):
    base = Path(base_data_dir)
    meta_path = base / "indices" / "faiss" / "index_meta.json"

    if not meta_path.exists():
        return {}

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
