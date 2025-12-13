from src.common.retriever.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever()

    pregunta = "What is SparseSwaps?"

    resultados = retriever.retrieve(pregunta, k=5)

    print("\n=== RESULTADOS DEL RETRIEVER ===\n")

    for i, r in enumerate(resultados, 1):
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"Documento: {r['doc_id']}")
        print(f"Pagina: {r['page']}")
        print(f"Fragmento: {r['frag_id']}")
        print(r["text"][:300])
        print("-" * 50)
