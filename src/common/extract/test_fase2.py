import json
from pathlib import Path

# Tomar el primer archivo jsonl del texto limpio
archivo = next(Path("data/preprocessed").glob("*.jsonl"))

with open(archivo, "r", encoding="utf-8") as f:
    linea = json.loads(next(f))

assert "clean_text" in linea
assert len(linea["clean_text"]) > 50

print("Fase 2 validada correctamente")
