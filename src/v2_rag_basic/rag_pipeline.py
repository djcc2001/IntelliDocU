# src/v2_rag_basic/rag_pipeline.py
from src.common.retriever.retriever import Retriever
from src.v2_rag_basic.prompt import build_prompt

class RAGPipeline:
    def __init__(self, llm):
        self.retriever = Retriever()
        self.llm = llm

    def answer(self, question, k=5, max_fragment_length=400):
        fragments = self.retriever.retrieve(question, k=k)
        
        # Truncar fragmentos muy largos
        for frag in fragments:
            if len(frag["text"]) > max_fragment_length:
                frag["text"] = frag["text"][:max_fragment_length] + " ..."

        prompt = build_prompt(question, fragments)
        response = self.llm.generate(prompt).strip()
        return {"question": question, "answer": response, "fragments": fragments}
