"""
Pipeline RAG Advanced balanceado: menos abstenciones en preguntas válidas,
manteniendo seguridad frente a preguntas imposibles.
"""

from src.common.retriever.retriever import Retriever
from src.v3_rag_advanced.config import (
    TOP_K,
    MAX_FRAGMENTS,
    SCORE_MINIMO,
    ABSTENTION_TEXT
)
from src.v3_rag_advanced.context_builder import build_limited_context
from src.v3_rag_advanced.prompt import build_prompt, format_answer_with_citations


class RAGAdvancedPipeline:
    """
    Pipeline RAG con citación y abstención balanceada.
    """

    def __init__(
        self,
        llm,
        top_k=TOP_K,
        max_fragments=MAX_FRAGMENTS,
        score_minimo=SCORE_MINIMO
    ):
        self.retriever = Retriever()
        self.llm = llm
        self.top_k = top_k
        self.max_fragments = max_fragments
        self.score_minimo = score_minimo

    # --------------------------------------------------
    # Early abstention balanceada
    # --------------------------------------------------
    def early_abstain(self, question, fragmentos):
        # 1) Sin fragmentos → abstención inmediata
        if not fragmentos:
            return True

        # 2) Todos los fragmentos con score muy bajo → abstención
        scores = [f.get("score", 0.0) for f in fragmentos]
        if max(scores) < self.score_minimo:
            return True

        # 3) Revisión de palabras clave de la pregunta
        keywords = [w.lower() for w in question.split() if len(w) > 3]
        if len(keywords) < 2:
            return False  # pregunta muy corta → no abstener

        context_text = " ".join(f.get("text", "").lower() for f in fragmentos)
        missing = [w for w in keywords if w not in context_text]

        # Threshold balanceado: 70% de palabras faltantes → abstenerse
        if len(missing) / max(len(keywords), 1) > 0.7:
            return True

        return False

    # --------------------------------------------------
    # Hallucination check post-LLM (estricto)
    # --------------------------------------------------
    def hallucination_check(self, answer, fragmentos):
        if not fragmentos:
            return True  # sin fragmentos → abstención
        context_text = " ".join(f.get("text", "").lower() for f in fragmentos)
        answer_text = answer.lower().replace(ABSTENTION_TEXT.lower(), "")
        for w in answer_text.split():
            if len(w) > 3 and w not in context_text:
                return True
        return False

    # --------------------------------------------------
    # Método principal de respuesta
    # --------------------------------------------------
    def answer(self, question):
        fragmentos = self.retriever.retrieve(question, k=self.top_k)

        # 1) Early abstention
        if self.early_abstain(question, fragmentos):
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": [],
                "abstained": True
            }

        # 2) Construir contexto limitado
        context, fragmentos_usados = build_limited_context(
            fragmentos, max_fragments=self.max_fragments
        )

        # 3) Generar respuesta
        prompt = build_prompt(context, question)
        respuesta = self.llm.generate(prompt)
        respuesta = respuesta.strip() if respuesta else ABSTENTION_TEXT

        # 4) Hallucination check post-LLM
        if respuesta != ABSTENTION_TEXT and self.hallucination_check(respuesta, fragmentos_usados):
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": [],
                "abstained": True
            }

        # 5) Añadir citaciones si hay respuesta válida
        if respuesta == ABSTENTION_TEXT:
            return {
                "question": question,
                "answer": respuesta,
                "fragments": [],
                "abstained": True
            }
        else:
            respuesta_con_citas = format_answer_with_citations(respuesta, fragmentos_usados)
            return {
                "question": question,
                "answer": respuesta_con_citas,
                "fragments": fragmentos_usados,
                "abstained": False
            }
