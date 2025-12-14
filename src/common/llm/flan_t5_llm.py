# src/common/llm/flan_t5_llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FlanT5LLM:
    def __init__(self, model_name="google/flan-t5-base", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def generate(self, prompt: str, max_length=256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False  # determinista, más coherente para RAG
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
