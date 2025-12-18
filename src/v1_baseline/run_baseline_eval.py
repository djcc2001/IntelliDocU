# run_baseline.py
import random
from src.common.llm.qwen_llm import QwenLLM

SEED = 42

def build_prompt(question):
    system_prompt = (
        "You are an academic assistant.\n"
        "You must answer based ONLY on your general knowledge.\n"
        "If you are not certain that the information is correct, "
        "explicitly say that you do not know.\n"
        "Do NOT invent details.\n"
        "Do NOT assume the contents of any specific document.\n"
        "Be concise and factual."
    )

    user_prompt = (
        f"Question:\n{question}\n\n"
        "Answer (or state that the information is unknown):"
    )

    return system_prompt, user_prompt


def run_baseline(question):
    random.seed(SEED)
    system_prompt, user_prompt = build_prompt(question)

    llm = QwenLLM()
    prompt = f"{system_prompt}\n\n{user_prompt}"

    response = llm.generate(prompt).strip()
    return response


if __name__ == "__main__":
    question = "Does DuetSVG implement a reinforcement learning module for path optimization?"
    answer = run_baseline(question)
    print("\n=== BASELINE RESPONSE ===\n")
    print(answer)
