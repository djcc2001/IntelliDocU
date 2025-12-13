import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Configuracion general
INPUT_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("data/fragments")

CHUNK_SIZE = 500
OVERLAP = 150

# Crear carpeta de salida si no existe
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

def chunk_text(text):
    """
    Divide un texto en fragmentos usando tokens con overlap
    """
    tokens = tokenizer.tokenize(text)
    chunks = []

    start = 0
    frag_id = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)

        chunks.append({
            "frag_id": frag_id,
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })

        frag_id += 1

        if end == len(tokens):
            break

        start = end - OVERLAP

    return chunks


def process_file(jsonl_path):
    """
    Procesa un archivo jsonl de texto limpio y genera chunks
    """
    doc_id = jsonl_path.stem
    output_path = OUTPUT_DIR / f"{doc_id}_fragments.jsonl"

    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)

            clean_text = data.get("clean_text", "").strip()
            page = data.get("page", None)

            if not clean_text:
                continue

            chunks = chunk_text(clean_text)

            for chunk in chunks:
                record = {
                    "doc_id": doc_id,
                    "page": page,
                    "frag_id": chunk["frag_id"],
                    "text": chunk["text"],
                    "token_count": chunk["token_count"]
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Chunks generados: {doc_id}")


def main():
    files = list(INPUT_DIR.glob("*.jsonl"))

    if not files:
        print("No se encontraron archivos preprocesados")
        return

    for file_path in tqdm(files, desc="Procesando documentos"):
        process_file(file_path)


if __name__ == "__main__":
    main()
