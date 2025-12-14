import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import re

# ===============================
# Configuración general
# ===============================
INPUT_DIR = Path("data/preprocessed")
OUTPUT_DIR = Path("data/fragments")
CHUNK_SIZE = 512       # tokens por chunk
OVERLAP = 100          # tokens superpuestos
MIN_CHUNK_TOKENS = 50  # evita chunks finales muy pequeños

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tokenizer Flan-T5
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# ===============================
# Patrones de secciones
# ===============================
import re

SECTION_PATTERNS = [
    # Appendix primero para no confundir
    (r"^\s*(appendix|a\.1|a\.)\b", "appendix"),

    # Abstract
    (r"^\s*abstract\b", "abstract"),

    # Introduction
    (r"^\s*1\s*\.?\s*introduction\b", "introduction"),
    (r"^\s*introduction\b", "introduction"),

    # Method / Algorithm
    (r"^\s*2\s*\.?\s*(method|algorithm|approach|sparse)", "method"),
    (r"^\s*method\b", "method"),
    (r"^\s*algorithm\b", "method"),
    (r"^\s*approach\b", "method"),

    # Results / Experiments
    (r"^\s*(experiment|result|evaluation|study)\b", "results"),

    # Conclusion (opcional)
    (r"^\s*conclusion\b", "conclusion"),
]

def detect_section(text):
    """
    Detecta la sección de un fragmento usando regex jerárquico.
    Primero appendix, luego abstract, introduction, method, results, conclusion.
    """
    head = text[:1000].lower()  # Mirar los primeros 1000 caracteres
    for pattern, section in SECTION_PATTERNS:
        if re.search(pattern, head):
            return section
    return "unknown"


# ===============================
# Funciones de chunking
# ===============================
def chunk_text(text):
    """Divide un texto en fragmentos usando tokens con overlap."""
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_tensors=None
    )["input_ids"]

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]

        if len(chunk_tokens) < MIN_CHUNK_TOKENS:
            break

        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens)
        })

        if end == len(tokens):
            break

        start = end - OVERLAP

    return chunks

def process_file(jsonl_path):
    """Procesa un archivo jsonl preprocesado y genera chunks."""
    doc_id = jsonl_path.stem
    output_path = OUTPUT_DIR / f"{doc_id}_fragments.jsonl"
    global_frag_id = 0

    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            clean_text = data.get("clean_text", "").strip()
            page = data.get("page", None)

            # Detectar sección: usa la del preprocesamiento si existe
            section = data.get("section")
            if not section:
                section = detect_section(clean_text)

            if not clean_text:
                continue

            chunks = chunk_text(clean_text)

            for chunk in chunks:
                record = {
                    "doc_id": doc_id,
                    "page": page,
                    "section": section,
                    "frag_id": global_frag_id,
                    "text": chunk["text"],
                    "token_count": chunk["token_count"]
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_frag_id += 1

    print(f"Chunks generados para: {doc_id}")

def main():
    files = list(INPUT_DIR.glob("*.jsonl"))
    if not files:
        print("No se encontraron archivos preprocesados")
        return

    for file_path in tqdm(files, desc="Procesando documentos"):
        process_file(file_path)

if __name__ == "__main__":
    main()
