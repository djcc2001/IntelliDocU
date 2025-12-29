"""Script para probar RAG Advanced con una pregunta."""

from src.common.llm.qwen_llm import QwenLLM
#from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v3_rag_advanced.rag_pipeline import RAGAdvancedPipeline


def main():
    """Prueba el sistema con una pregunta de ejemplo."""
    
    # Pregunta de prueba
    question = (
        "Does DuetSVG implement a reinforcement learning module for path optimization?"
    )
    
    # Inicializar sistema
    print("Inicializando RAG Advanced...")
    llm = QwenLLM()
    rag = RAGAdvancedPipeline(llm)
    
    # Generar respuesta
    print(f"\nPregunta: {question}\n")
    result = rag.responder(question)
    
    # Mostrar resultado
    print("="*60)
    print("RESPUESTA:")
    print("="*60)
    print(result["answer"])
    print("\n" + "="*60)
    
    if result["abstained"]:
        print("⚠️  El sistema se abstuvo de responder")
    else:
        print(f"✓ Usado {len(result['fragments'])} fragmento(s)")


if __name__ == "__main__":
    main()
