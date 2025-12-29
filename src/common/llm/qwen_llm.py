"""
Modulo para el modelo de lenguaje Qwen.
Implementa la interfaz para generar respuestas usando el modelo Qwen2.5.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ModeloQwen:
    """
    Clase para interactuar con el modelo de lenguaje Qwen.
    """
    
    def __init__(self, nombre_modelo="Qwen/Qwen2.5-1.5B-Instruct", dispositivo="auto"):
        """
        Inicializa el modelo Qwen.
        
        Args:
            nombre_modelo: Nombre del modelo a cargar desde HuggingFace
            dispositivo: "auto", "cuda", o "cpu"
        """
        print(f"Cargando modelo {nombre_modelo}...")

        # Determinar dispositivo automaticamente si es necesario
        if dispositivo == "auto":
            self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.dispositivo = dispositivo

        # Configurar tipo de datos segun dispositivo
        tipo_datos = torch.float16 if self.dispositivo == "cuda" else torch.float32

        # Cargar tokenizer primero
        print("Cargando tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            nombre_modelo,
            trust_remote_code=True
        )

        # Configurar token de padding si no existe
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cargar modelo optimizado
        print("Cargando modelo...")
        self.modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            dtype=tipo_datos,  # Usar dtype en lugar de torch_dtype (deprecated)
            device_map="auto" if self.dispositivo == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True  # Importante para ahorrar memoria
        )

        # Mover a dispositivo si no esta en cuda con device_map
        if self.dispositivo != "cuda":
            self.modelo = self.modelo.to(self.dispositivo)

        print(f"Modelo cargado en {self.dispositivo}")
    
    def generar(self, prompt, max_tokens_nuevos=512):
        """
        Genera respuesta para un prompt.
        
        Args:
            prompt: Texto del prompt (puede ser string o tupla (system, user))
            max_tokens_nuevos: Maximo de tokens a generar
        
        Returns:
            Respuesta generada como string
        """
        # Formatear el prompt para Qwen 2.5 (formato instruct)
        if isinstance(prompt, tuple):
            # Si es tupla (system, user), formatear apropiadamente
            system_prompt, user_prompt = prompt
            prompt_formateado = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Si es string simple, usar formato instruct basico
            prompt_formateado = f"<|im_start|>system\nEres un asistente util.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenizar
        entradas = self.tokenizer(
            prompt_formateado,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.dispositivo)

        # Configuracion de generacion (determinista, sin sampling)
        configuracion_generacion = {
            'input_ids': entradas.input_ids,
            'attention_mask': entradas.attention_mask,
            'max_new_tokens': max_tokens_nuevos,
            'do_sample': False,  # Determinista para RAG
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        # Generar respuesta (usar generate() en lugar de generar())
        with torch.no_grad():
            salidas = self.modelo.generate(**configuracion_generacion)

        # Decodificar solo los tokens nuevos (excluir el prompt)
        respuesta = self.tokenizer.decode(
            salidas[0][entradas.input_ids.shape[1]:], 
            skip_special_tokens=True
        )

        return respuesta.strip()


# Alias para mantener compatibilidad con codigo existente
QwenLLM = ModeloQwen
