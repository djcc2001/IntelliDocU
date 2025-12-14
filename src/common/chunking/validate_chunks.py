import json
from pathlib import Path
from statistics import mean

FRAGMENTS_DIR = Path("data/fragments")

# Deben coincidir con el chunker
MIN_TOKENS = 50
MAX_TOKENS = 512

def validate_file(path):
    """
    Valida un archivo de fragments y devuelve estadísticas
    """
    token_counts = []
    empty_chunks = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "").strip()
            tokens = data.get("token_count", 0)

            if not text or tokens == 0:
                empty_chunks += 1
            else:
                token_counts.append(tokens)

    return {
        "total_chunks": len(token_counts) + empty_chunks,
        "empty_chunks": empty_chunks,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "avg_tokens": round(mean(token_counts), 2) if token_counts else 0
    }

def main():
    files = list(FRAGMENTS_DIR.glob("*_fragments.jsonl"))
    if not files:
        print("No se encontraron archivos de fragments")
        return

    print("\n=== VALIDACIÓN DE CHUNKS ===\n")

    for file_path in files:
        doc_id = file_path.stem.replace("_fragments", "")
        stats = validate_file(file_path)

        print(f"Documento: {doc_id}")
        print(f"  Total chunks : {stats['total_chunks']}")
        print(f"  Chunks vacíos: {stats['empty_chunks']}")
        print(f"  Tokens min  : {stats['min_tokens']}")
        print(f"  Tokens max  : {stats['max_tokens']}")
        print(f"  Tokens prom : {stats['avg_tokens']}")

        if stats["empty_chunks"] > 0:
            print("  ADVERTENCIA: existen chunks vacíos")

        if stats["min_tokens"] < MIN_TOKENS:
            print("  Nota: existen chunks pequeños")

        if stats["max_tokens"] > MAX_TOKENS:
            print("  Nota: existen chunks más grandes de lo esperado")

        print()

    print("=== FIN VALIDACIÓN ===")

if __name__ == "__main__":
    main()
