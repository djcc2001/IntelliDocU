"""
Pipeline RAG Advanced (modo producción, con control estricto de abstención).
"""

import re
from src.common.retriever.retriever import Retriever
from src.v3_rag_advanced.config import (
    TOP_K,
    MAX_FRAGMENTS,
    ABSTENTION_TEXT
)
from src.v3_rag_advanced.context_builder import build_limited_context
from src.v3_rag_advanced.prompt import (
    build_prompt,
    format_answer_with_citations
)


class RAGAdvancedPipeline:
    """
    Pipeline RAG con citación y abstención robusta (anti-hallucination).
    """

    def __init__(
        self,
        llm,
        top_k=TOP_K,
        max_fragments=MAX_FRAGMENTS
    ):
        self.retriever = Retriever()
        self.llm = llm
        self.top_k = top_k
        self.max_fragments = max_fragments

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def evidence_strength(self, fragmentos):
        """
        Evalúa fuerza de evidencia para cosine similarity.
        Score ∈ [0,1], mayor es mejor.
        """
        if not fragmentos:
            return "none"

        top_score = fragmentos[0].get("score", 0.0)

        if top_score >= 0.6:
            return "strong"
        if top_score >= 0.4:
            return "weak"
        return "none"

    def answer_uses_context(self, answer, fragmentos):
        """
        Verifica si la respuesta reutiliza términos del contexto
        (anti-definiciones genéricas).
        """
        answer = answer.lower()

        context_text = " ".join(
            f["text"].lower() for f in fragmentos
        )

        keywords = set(
            w for w in re.findall(r"\b[a-zA-Z]{5,}\b", context_text)
        )

        hits = sum(1 for k in keywords if k in answer)
        return hits >= 1

    def should_abstain_early(self, question):
        q = question.lower().strip()

        if len(q.split()) < 2:
            return True

        greetings = {"hi", "hello", "hola", "thanks", "gracias"}
        if q in greetings:
            return True

        return False


    # ------------------------------------------------------------------
    # PIPELINE PRINCIPAL
    # ------------------------------------------------------------------

    def answer(self, question):
        # 1️⃣ Abstención temprana
        if self.should_abstain_early(question):
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": [],
                "abstained": True
            }

        # 2️⃣ Recuperación
        fragmentos = self.retriever.retrieve(question, k=self.top_k)

        if not fragmentos:
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": [],
                "abstained": True
            }

        # 3️⃣ Evaluar evidencia
        strength = self.evidence_strength(fragmentos)

        if strength == "none":
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": [],
                "abstained": True
            }

        # 4️⃣ Construcción de contexto
        context, usados = build_limited_context(
            fragmentos,
            max_fragments=self.max_fragments
        )

        # 5️⃣ Prompt + generación
        prompt = build_prompt(context, question)
        respuesta = self.llm.generate(prompt)
        respuesta = respuesta.strip() if respuesta else ""

        # 6️⃣ Abstención post-LLM
        if not respuesta or respuesta == ABSTENTION_TEXT:
            return {
                "question": question,
                "answer": ABSTENTION_TEXT,
                "fragments": usados,
                "abstained": True
            }

        # 8️⃣ Respuesta final con citas
        respuesta_final = format_answer_with_citations(
            respuesta,
            usados
        )

        return {
            "question": question,
            "answer": respuesta_final,
            "fragments": usados,
            "abstained": False
        }
