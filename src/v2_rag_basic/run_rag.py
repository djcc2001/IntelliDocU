# src/v2_rag_basic/run_rag.py

#from src.common.llm.qwen_llm import QwenLLM
from src.common.llm.flan_t5_llm import FlanT5LLM
from src.v2_rag_basic.rag_pipeline import RAGPipeline

def main():
    question = "Does DuetSVG implement a reinforcement learning module for path optimization?"
    llm = FlanT5LLM()
    rag = RAGPipeline(llm)

    result = rag.answer(question)

    print("\n=== RAG V2 RESPONSE ===\n")
    print(result["answer"])

if __name__ == "__main__":
    main()
