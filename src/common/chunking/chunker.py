import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# ===============================
# Configuración RAG-aware
# ===============================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 384      # Óptimo para MiniLM
OVERLAP = 64          # Overlap semántico
MIN_CHUNK_TOKENS = 50

print("Cargando tokenizer de embeddings...")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)

# -------------------------------------------------
# Chunking basado en tokens de EMBEDDINGS
# -------------------------------------------------
def chunk_tokens(tokens, chunk_size, overlap):
    start = 0
    chunks = []

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]

        if len(chunk) < MIN_CHUNK_TOKENS and start > 0:
            break

        chunks.append((start, end, chunk))

        if end == len(tokens):
            break

        start = end - overlap

    return chunks

def process_file(jsonl_path, output_dir):
    doc_id = jsonl_path.stem
    output_path = output_dir / f"{doc_id}_fragments.jsonl"

    frag_id = 0
    pages = set()
    chunks_generados = 0

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        # 🔑 Agrupar por sección para continuidad semántica
        secciones = {}

        for line in fin:
            data = json.loads(line)
            section = data.get("section", "unknown")
            secciones.setdefault(section, []).append(data)

        for section, items in secciones.items():
            # Unir texto de la sección (cruza páginas)
            texto = "\n\n".join(i["clean_text"] for i in items if i.get("clean_text"))
            if not texto:
                continue

            pages_section = sorted({i.get("page") for i in items})
            pages.update(pages_section)

            tokens = tokenizer(
                texto,
                add_special_tokens=False,
                truncation=False
            )["input_ids"]

            token_chunks = chunk_tokens(tokens, CHUNK_SIZE, OVERLAP)

            for idx, (start, end, chunk_ids) in enumerate(token_chunks):
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

                record = {
                    "doc_id": doc_id,
                    "section": section,
                    "pages": pages_section,
                    "frag_id": frag_id,
                    "chunk_in_section": idx,
                    "text": chunk_text,
                    "token_count": len(chunk_ids)
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                frag_id += 1
                chunks_generados += 1

    print(f"  ✓ {doc_id}: {len(pages)} páginas → {chunks_generados} chunks")


def generar_chunks(base_data_dir="data"):
    base = Path(base_data_dir)
    input_dir = base / "preprocessed"
    output_dir = base / "fragments"

    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.jsonl"))
    if not files:
        print("❌ No hay archivos preprocessed")
        return

    print("\nIniciando Fase 3: Chunking (RAG-aware)")
    print(f"  Embeddings: {EMBEDDING_MODEL}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Overlap: {OVERLAP}\n")

    for f in tqdm(files, desc="Chunking"):
        process_file(f, output_dir)

    total = sum(
        1 for f in output_dir.glob("*_fragments.jsonl")
        for _ in open(f, encoding="utf-8")
    )

    print("\n✓ Fase 3 completada")
    print(f"  Total chunks: {total}")
    print(f"  Output: {output_dir}")

if __name__ == "__main__":
    generar_chunks()
