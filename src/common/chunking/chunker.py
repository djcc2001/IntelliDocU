import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# ===============================
# Configuración
# ===============================
CHUNK_SIZE = 512  # Tokens por chunk
OVERLAP = 100     # Tokens de overlap entre chunks
MIN_CHUNK_TOKENS = 50  # Mínimo para considerar un chunk válido

# Inicializar tokenizer
print("Cargando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP, min_tokens=MIN_CHUNK_TOKENS):
    """
    Divide el texto en chunks de tamaño fijo con overlap.
    
    Args:
        text: Texto a dividir
        chunk_size: Tamaño máximo en tokens
        overlap: Tokens de solapamiento
        min_tokens: Mínimo de tokens para considerar válido
    
    Returns:
        Lista de dicts con texto y conteo de tokens
    """
    # Tokenizar todo el texto
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    )["input_ids"]

    chunks = []
    start = 0

    while start < len(tokens):
        # Calcular fin del chunk actual
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Descartar chunks muy pequeños (excepto el último)
        if len(chunk_tokens) < min_tokens and start > 0:
            break

        # Decodificar tokens a texto
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        chunks.append({
            "text": chunk_text,
            "token_count": len(chunk_tokens),
            "start_token": start,
            "end_token": end
        })

        # Si llegamos al final, terminar
        if end == len(tokens):
            break

        # Avanzar con overlap
        start = end - overlap

    return chunks


def process_file(jsonl_path, output_dir):
    """
    Procesa un archivo JSONL y genera chunks.
    
    Args:
        jsonl_path: Ruta al archivo preprocessed
        output_dir: Directorio de salida para fragments
    """
    doc_id = jsonl_path.stem
    output_path = output_dir / f"{doc_id}_fragments.jsonl"
    
    global_frag_id = 0
    chunks_generados = 0
    paginas_procesadas = 0

    with open(jsonl_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            data = json.loads(line)
            
            # Extraer datos de la Fase 2
            clean_text = data.get("clean_text", "").strip()
            page = data.get("page")
            section = data.get("section", "unknown")  # Ya viene de Fase 2
            
            # Saltar páginas vacías
            if not clean_text:
                continue

            paginas_procesadas += 1
            
            # Generar chunks del texto limpio
            chunks = chunk_text(clean_text)

            # Guardar cada chunk
            for idx, chunk in enumerate(chunks):
                record = {
                    "doc_id": doc_id,
                    "page": page,
                    "section": section,
                    "frag_id": global_frag_id,
                    "chunk_in_page": idx,  # Posición del chunk dentro de la página
                    "text": chunk["text"],
                    "token_count": chunk["token_count"]
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                global_frag_id += 1
                chunks_generados += 1

    print(f"  ✓ {doc_id}: {paginas_procesadas} páginas → {chunks_generados} chunks")


def generar_chunks(base_data_dir="data"):
    """
    Ejecuta la Fase 3: Chunking completo del dataset.
    
    Args:
        base_data_dir: Directorio base del proyecto
    """
    base_data_dir = Path(base_data_dir)
    input_dir = base_data_dir / "preprocessed"
    output_dir = base_data_dir / "fragments"

    # Crear directorio de salida
    output_dir.mkdir(parents=True, exist_ok=True)

    # Buscar archivos preprocessed
    files = list(input_dir.glob("*.jsonl"))
    
    if not files:
        print("❌ No se encontraron archivos en data/preprocessed/")
        print("   Ejecuta primero la Fase 2 (cleaner.py)")
        return

    print(f"\nIniciando Fase 3: Chunking")
    print(f"Configuración:")
    print(f"  - Chunk size: {CHUNK_SIZE} tokens")
    print(f"  - Overlap: {OVERLAP} tokens")
    print(f"  - Mínimo: {MIN_CHUNK_TOKENS} tokens")
    print(f"\nProcesando {len(files)} archivo(s)...\n")

    # Procesar cada archivo
    for file_path in tqdm(files, desc="Chunking"):
        process_file(file_path, output_dir)

    print(f"\n✓ Fase 3 completada")
    print(f"  Archivos generados en: {output_dir}")
    
    # Estadísticas finales
    total_chunks = sum(1 for f in output_dir.glob("*_fragments.jsonl") 
                      for _ in open(f, encoding="utf-8"))
    print(f"  Total de chunks: {total_chunks}")


if __name__ == "__main__":
    generar_chunks()