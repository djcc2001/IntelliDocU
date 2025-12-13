# src/common/llm/simple_llm.py

class SimpleLLM:
    def __init__(self):
        pass

    def generate(self, prompt: str) -> str:
        """
        LLM simulado para pruebas RAG.
        Devuelve el contexto recuperado para inspeccion.
        """
        return prompt
