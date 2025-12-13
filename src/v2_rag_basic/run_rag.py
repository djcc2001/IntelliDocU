# src/v2_rag_basic/run_rag.py

from src.common.llm.simple_llm import SimpleLLM
from src.v2_rag_basic.rag_pipeline import RAGPipeline

if __name__ == "__main__":

    llm = SimpleLLM()
    rag = RAGPipeline(llm)

    question = "What is SparseSwaps?"
    result = rag.answer(question)

    print("\n=== RAG BASIC RESPONSE ===\n")
    print(result["answer"])
