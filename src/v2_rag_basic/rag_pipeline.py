# src/v2_rag_basic/rag_pipeline.py

from src.common.retriever.retriever import Retriever
from src.v2_rag_basic.prompt import build_prompt

class RAGPipeline:

    def __init__(self, llm):
        self.retriever = Retriever()
        self.llm = llm

    def answer(self, question, k=5):
        fragments = self.retriever.retrieve(question, k=k)
        prompt = build_prompt(question, fragments)
        response = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": response,
            "fragments": fragments
        }
