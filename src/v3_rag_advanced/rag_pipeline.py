import re
from src.common.retriever.retriever import Retriever
from src.v3_rag_advanced.config import TOP_K, MAX_FRAGMENTS, ABSTENTION_TEXT
from src.v3_rag_advanced.context_builder import build_limited_context
from src.v3_rag_advanced.prompt import build_prompt, format_answer_with_citations


class RAGAdvancedPipeline:
    def __init__(self, llm, base_data_dir="data"):
        self.retriever = Retriever(base_data_dir=base_data_dir)
        self.llm = llm
        self.top_k = TOP_K
        self.max_fragments = MAX_FRAGMENTS

    def evidence_strength(self, fragments):
        if not fragments:
            return "none"

        s = fragments[0]["score"]
        if s >= 0.30:
            return "strong"
        if s >= 0.18:
            return "weak"
        return "none"

    def answer_uses_context(self, answer, fragments):
        answer = answer.lower()
        context_text = " ".join(f["text"].lower() for f in fragments)

        keywords = set(re.findall(r"\b[a-zA-Z]{5,}\b", context_text))
        return any(k in answer for k in keywords)

    def should_abstain_early(self, question):
        q = question.lower().strip()
        if len(q.split()) < 2:
            return True
        if q in {"hi", "hello", "hola", "thanks", "gracias"}:
            return True
        return False

    def answer(self, question):
        if self.should_abstain_early(question):
            return self._abstain(question)

        fragments = self.retriever.retrieve(
            question,
            k=self.top_k,
            min_score=0.18
        )

        if self.evidence_strength(fragments) == "none":
            return self._abstain(question)

        context, usados = build_limited_context(
            fragments,
            max_fragments=self.max_fragments
        )

        prompt = build_prompt(context, question)
        raw_answer = (self.llm.generate(prompt) or "").strip()

        if not raw_answer or raw_answer == ABSTENTION_TEXT:
            return self._abstain(question, usados)

        if not self.answer_uses_context(raw_answer, usados):
            return self._abstain(question, usados)

        final = format_answer_with_citations(raw_answer, usados)

        return {
            "question": question,
            "answer": final,
            "fragments": usados,
            "abstained": False
        }

    def _abstain(self, question, fragments=None):
        return {
            "question": question,
            "answer": ABSTENTION_TEXT,
            "fragments": fragments or [],
            "abstained": True
        }
