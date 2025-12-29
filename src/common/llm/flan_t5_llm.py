"""
Modulo para el modelo de lenguaje Flan-T5.
Implementa la interfaz para generar respuestas usando el modelo Flan-T5 de Google.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ModeloFlanT5:
    """
    Clase para interactuar con el modelo de lenguaje Flan-T5.
    """
    
    def __init__(self, nombre_modelo="google/flan-t5-base", dispositivo="cpu"):
        """
        Inicializa el modelo Flan-T5.
        
        Args:
            nombre_modelo: Nombre del modelo a cargar desde HuggingFace
            dispositivo: Dispositivo donde cargar el modelo ("cpu" o "cuda")
        """
        self.dispositivo = dispositivo
        self.tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
        self.modelo = AutoModelForSeq2SeqLM.from_pretrained(nombre_modelo).to(dispositivo)

    def generar(self, prompt: str, longitud_maxima=256) -> str:
        """
        Genera respuesta para un prompt.
        
        Args:
            prompt: Texto del prompt
            longitud_maxima: Longitud maxima de la respuesta generada
        
        Returns:
            Respuesta generada como string
        """
        entradas = self.tokenizer(prompt, return_tensors="pt").to(self.dispositivo)
        salidas = self.modelo.generar(
            **entradas,
            max_length=longitud_maxima,
            do_sample=False  # Determinista, mas coherente para RAG
        )
        return self.tokenizer.decode(salidas[0], skip_special_tokens=True)


# Alias para mantener compatibilidad con codigo existente
FlanT5LLM = ModeloFlanT5
