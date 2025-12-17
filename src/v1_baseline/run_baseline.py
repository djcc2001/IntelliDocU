# run_baseline.py
import random
from src.common.llm.qwen_llm import QwenLLM
#from src.common.llm.flan_t5_llm import FlanT5LLM

SEED = 42

def build_prompt(question):
    system_prompt = (
        "You are a helpful academic assistant. "
        "Answer the question as clearly and accurately as possible."
    )
    user_prompt = f"Question: {question}\nAnswer:"
    return system_prompt, user_prompt

def run_baseline(question):
    random.seed(SEED)
    system_prompt, user_prompt = build_prompt(question)
    llm = QwenLLM()
    prompt = f"{system_prompt}\n{user_prompt}"
    response = llm.generate(prompt).strip()
    return response

if __name__ == "__main__":
    question = "Does DuetSVG implement a reinforcement learning module for path optimization?"
    answer = run_baseline(question)
    print("\n=== BASELINE RESPONSE ===\n")
    print(answer)
