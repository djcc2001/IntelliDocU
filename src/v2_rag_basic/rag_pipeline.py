# src/v2_rag_basic/rag_pipeline.py

from src.common.retriever.retriever import Retriever
from src.v2_rag_basic.prompt import (
    build_literal_context,
    build_partial_summary_prompt,
    MAX_CONTEXT_CHARS
)


class RAGPipeline:
    def __init__(self, llm, top_k=10, max_fragment_length=400):
        self.retriever = Retriever()
        self.llm = llm
        self.top_k = top_k
        self.max_fragment_length = max_fragment_length

    def answer(self, question):
        fragments = self.retriever.retrieve(question, k=self.top_k)

        if not fragments:
            return {
                "question": question,
                "answer": "The provided context does not contain enough information to answer the question.",
                "fragments": []
            }

        # Truncado defensivo
        for f in fragments:
            f["text"] = f["text"][: self.max_fragment_length]

        context = build_literal_context(fragments)

        # Corte duro de contexto total
        context = context[:MAX_CONTEXT_CHARS]

        prompt = build_partial_summary_prompt(context, question)
        answer = self.llm.generate(prompt).strip()

        return {
            "question": question,
            "answer": answer,
            "fragments": fragments
        }
