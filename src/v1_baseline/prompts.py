def build_prompt(question):
    system_prompt = (
        "You are a helpful academic assistant. "
        "Answer the question as clearly and accurately as possible."
    )

    user_prompt = f"Question: {question}\nAnswer:"

    return system_prompt, user_prompt
