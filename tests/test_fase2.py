import json
from pathlib import Path

# Proyecto root para rutas relativas
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"

# Tomar el primer archivo jsonl del texto limpio
archivo = next(DATA_DIR.glob("*.jsonl"))

with open(archivo, "r", encoding="utf-8") as f:
    linea = json.loads(next(f))

assert "clean_text" in linea
assert "section" in linea
assert len(linea["clean_text"]) > 50

print("Fase 2 validada correctamente")