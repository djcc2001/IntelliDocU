"""
Script de validacion del recuperador.
Permite probar la recuperacion de fragmentos con consultas de ejemplo.
"""

from src.common.retriever.retriever import Recuperador

if __name__ == "__main__":
    recuperador = Recuperador()

    pregunta = "What is SparseSwaps?"

    resultados = recuperador.recuperar(
        pregunta,
        k=5,
        secciones_permitidas=["abstract", "unknown"]
    )

    print("\n=== RESULTADOS DEL RECUPERADOR ===\n")

    for indice, resultado in enumerate(resultados, 1):
        print(f"[{indice}] Puntuacion: {resultado['score']:.4f}")
        print(f"Documento: {resultado['doc_id']}")
        print(f"Seccion : {resultado.get('section')}")
        print(f"Paginas  : {resultado.get('pages', resultado.get('page', '?'))}")
        print(f"Frag ID : {resultado['frag_id']}")
        print(resultado["text"][:300])
        print("-" * 50)
