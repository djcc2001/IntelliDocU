import random
from prompts import build_prompt
from config import TEMPERATURE, MAX_TOKENS, SEED

# Si usas OpenAI:
# from openai import OpenAI
# client = OpenAI()

def run_baseline(question):
    random.seed(SEED)

    system_prompt, user_prompt = build_prompt(question)

    # ---- EJEMPLO CON PSEUDO RESPUESTA ----
    # Reemplaza esto con tu llamada real al LLM
    response = (
        "SparseSwaps is a pruning refinement method for large language models "
        "that improves sparsity masks by iteratively swapping weights to "
        "reduce performance degradation."
    )

    return response


if __name__ == "__main__":
    question = "What is SparseSwaps?"
    answer = run_baseline(question)

    print("\n=== BASELINE RESPONSE ===\n")
    print(answer)
