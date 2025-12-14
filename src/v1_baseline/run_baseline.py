# run_baseline.py
import random
from src.v1_baseline.prompts import build_prompt
from src.v1_baseline.config import TEMPERATURE, MAX_TOKENS, SEED
from src.common.llm.flan_t5_llm import FlanT5LLM

def run_baseline(question):
    random.seed(SEED)
    system_prompt, user_prompt = build_prompt(question)
    llm = FlanT5LLM()
    prompt = f"{system_prompt}\n{user_prompt}"
    response = llm.generate(prompt).strip()
    return response

if __name__ == "__main__":
    question = "How does SPARSESWAPS improve pruning for LLMs compared to traditional magnitude pruning methods?"
    answer = run_baseline(question)
    print("\n=== BASELINE RESPONSE ===\n")
    print(answer)
