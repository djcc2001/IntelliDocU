from src.common.retriever.retriever import Retriever

if __name__ == "__main__":
    retriever = Retriever()

    pregunta = "What is SparseSwaps?"

    resultados = retriever.retrieve(
        pregunta,
        k=5,
        allowed_sections=["abstract", "unknown"]
    )

    print("\n=== RESULTADOS DEL RETRIEVER ===\n")

    for i, r in enumerate(resultados, 1):
        print(f"[{i}] Score: {r['score']:.4f}")
        print(f"Documento: {r['doc_id']}")
        print(f"Sección : {r.get('section')}")
        print(f"Página  : {r['page']}")
        print(f"Frag ID : {r['frag_id']}")
        print(r["text"][:300])
        print("-" * 50)
