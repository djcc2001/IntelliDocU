from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class QwenLLM:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct", device="auto"):
        """
        Args:
            model_name: Nombre del modelo a cargar
            device: "auto", "cuda", "cpu"
        """
        print(f"Loading model {model_name}...")

        # Determinar dispositivo
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Configurar dtype
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Cargar tokenizer primero
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Configurar padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cargar modelo optimizado
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Importante para ahorrar memoria
        )

        # Mover a dispositivo si no está en cuda con device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        print(f"Model loaded on {self.device}")
    
    def generate(self, prompt, max_new_tokens=512):
        """
        Genera respuesta para un prompt
        
        Args:
            prompt: Texto del prompt
            max_new_tokens: Máximo de tokens a generar
        """
        # Formatear el prompt para Qwen 2.5 (formato instruct)
        formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenizar
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        # Configuración de generación (determinista, sin sampling)
        generation_config = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': max_new_tokens,
            'do_sample': False,          
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        # Generar
        with torch.no_grad():
            outputs = self.model.generate(**generation_config)

        # Decodificar
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        return response.strip()
