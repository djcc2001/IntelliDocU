# src/v2_rag_basic/rag_pipeline.py

import faiss
import pickle
from pathlib import Path
from src.common.retriever.retriever import Retriever
from src.v2_rag_basic.prompt import (
    build_literal_context,
    build_partial_summary_prompt,
    MAX_CONTEXT_CHARS
)


class RAGPipeline:
    def __init__(self, llm, base_data_dir="data", top_k=10, max_fragment_length=400, faiss_index=None, texts=None):
        """
        llm: instancia del modelo LLM
        base_data_dir: carpeta donde están los índices y textos
        top_k: cuántos fragmentos recuperar
        max_fragment_length: límite de caracteres por fragmento
        faiss_index: índice FAISS precargado (opcional)
        texts: lista de textos correspondiente al índice (opcional)
        """
        self.llm = llm
        self.top_k = top_k
        self.max_fragment_length = max_fragment_length
        self.base_data_dir = Path(base_data_dir)

        # 🔹 Cargar FAISS y textos
        self.index = faiss_index
        self.texts = texts or []
        if self.index is None or not self.texts:
            self.index, self.texts = self._load_or_create_index()

        # 🔹 Inicializar retriever
        self.retriever = Retriever(base_data_dir=self.base_data_dir)

    def _load_or_create_index(self):
        """
        Carga índice FAISS y textos desde disco si existen,
        o devuelve un índice vacío y lista vacía.
        """
        index_path = self.base_data_dir / "indices/faiss/index.faiss"
        texts_path = self.base_data_dir / "indices/faiss/texts.pkl"

        if index_path.exists() and texts_path.exists():
            index = faiss.read_index(str(index_path))
            with open(texts_path, "rb") as f:
                texts = pickle.load(f)
            print(f"✅ FAISS index and texts loaded: {len(texts)} entries")
        else:
            index = None
            texts = []
            print("⚠️ FAISS index or texts not found, starting empty")

        return index, texts

    def answer(self, question: str) -> dict:
        """
        Recupera fragmentos relevantes y genera respuesta usando LLM.
        """
        fragments = self.retriever.retrieve(question, k=self.top_k)

        if not fragments:
            return {
                "question": question,
                "answer": "The provided context does not contain enough information to answer the question.",
                "fragments": []
            }

        # Truncado defensivo de fragmentos
        for f in fragments:
            f["text"] = f["text"][:self.max_fragment_length]

        # Construir contexto literal y limitar longitud total
        context = build_literal_context(fragments)
        context = context[:MAX_CONTEXT_CHARS]

        # Generar prompt y obtener respuesta
        prompt = build_partial_summary_prompt(context, question)
        answer = self.llm.generate(prompt).strip()

        return {
            "question": question,
            "answer": answer,
            "fragments": fragments
        }
