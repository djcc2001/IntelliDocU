"""
Pipeline de RAG Avanzado (Version 3).
Incluye verificacion de evidencia, abstención y citaciones de fuentes.
"""

import re
from src.common.retriever.retriever import Recuperador
from src.v3_rag_advanced.config import TOP_K, MAX_FRAGMENTOS, TEXTO_ABSTENCION
from src.v3_rag_advanced.context_builder import construir_contexto_limitatado
from src.v3_rag_advanced.prompt import construir_prompt, formatear_respuesta_con_citaciones


class PipelineRAGAvanzado:
    """
    Pipeline de RAG avanzado con verificacion de evidencia y abstención.
    """
    
    def __init__(self, modelo_llm, directorio_base_datos="data"):
        """
        Inicializa el pipeline RAG avanzado.
        
        Args:
            modelo_llm: Instancia del modelo LLM
            directorio_base_datos: Directorio base donde estan los datos
        """
        self.recuperador = Recuperador(directorio_base_datos=directorio_base_datos)
        self.modelo_llm = modelo_llm
        self.top_k = TOP_K
        self.max_fragmentos = MAX_FRAGMENTOS

    def fuerza_evidencia(self, fragmentos):
        """
        Evalua la fuerza de la evidencia basandose en la puntuacion del mejor fragmento.
        
        Args:
            fragmentos: Lista de fragmentos recuperados
        
        Returns:
            "strong", "weak", o "none" segun la fuerza de la evidencia
        """
        if not fragmentos:
            return "none"

        puntuacion = fragmentos[0]["score"]
        if puntuacion >= 0.30:
            return "strong"
        if puntuacion >= 0.18:
            return "weak"
        return "none"

    def respuesta_usa_contexto(self, respuesta, fragmentos):
        """
        Verifica si la respuesta usa palabras clave del contexto.
        
        Args:
            respuesta: Respuesta generada por el LLM
            fragmentos: Fragmentos usados como contexto
        
        Returns:
            True si la respuesta parece usar el contexto, False en caso contrario
        """
        respuesta_minusculas = respuesta.lower()
        texto_contexto = " ".join(fragmento["text"].lower() for fragmento in fragmentos)

        # Extraer palabras clave (palabras de 5+ caracteres)
        palabras_clave = set(re.findall(r"\b[a-zA-Z]{5,}\b", texto_contexto))
        return any(palabra_clave in respuesta_minusculas for palabra_clave in palabras_clave)

    def debe_abstener_temprano(self, pregunta):
        """
        Verifica si se debe abstenerse antes de procesar (preguntas muy cortas o saludos).
        
        Args:
            pregunta: Pregunta del usuario
        
        Returns:
            True si se debe abstenerse, False en caso contrario
        """
        pregunta_minusculas = pregunta.lower().strip()
        if len(pregunta_minusculas.split()) < 2:
            return True
        if pregunta_minusculas in {"hi", "hello", "hola", "thanks", "gracias"}:
            return True
        return False

    def responder(self, pregunta):
        """
        Genera respuesta usando RAG avanzado con verificacion de evidencia.
        
        Args:
            pregunta: Pregunta del usuario
        
        Returns:
            Diccionario con:
            - question: Pregunta original
            - answer: Respuesta generada o texto de abstención
            - fragments: Fragmentos usados
            - abstained: True si se abstuvo, False en caso contrario
        """
        # Abstención temprana para preguntas invalidas
        if self.debe_abstener_temprano(pregunta):
            return self._abstenerse(pregunta)

        # Recuperar fragmentos relevantes
        fragmentos = self.recuperador.recuperar(
            pregunta,
            k=self.top_k,
            puntuacion_minima=0.18
        )

        # Verificar fuerza de evidencia
        if self.fuerza_evidencia(fragmentos) == "none":
            return self._abstenerse(pregunta)

        # Construir contexto limitado
        contexto, fragmentos_usados = construir_contexto_limitatado(
            fragmentos,
            max_fragmentos=self.max_fragmentos
        )

        # Generar respuesta
        prompt = construir_prompt(contexto, pregunta)
        respuesta_cruda = (self.modelo_llm.generar(prompt) or "").strip()

        # Verificar si la respuesta es valida
        if not respuesta_cruda or respuesta_cruda == TEXTO_ABSTENCION:
            return self._abstenerse(pregunta, fragmentos_usados)

        # Verificar si la respuesta usa el contexto
        if not self.respuesta_usa_contexto(respuesta_cruda, fragmentos_usados):
            return self._abstenerse(pregunta, fragmentos_usados)

        # Formatear respuesta final con citaciones
        respuesta_final = formatear_respuesta_con_citaciones(respuesta_cruda, fragmentos_usados)

        return {
            "question": pregunta,
            "answer": respuesta_final,
            "fragments": fragmentos_usados,
            "abstained": False
        }

    def _abstenerse(self, pregunta, fragmentos=None):
        """
        Genera respuesta de abstención.
        
        Args:
            pregunta: Pregunta original
            fragmentos: Fragmentos recuperados (opcional)
        
        Returns:
            Diccionario con respuesta de abstención
        """
        return {
            "question": pregunta,
            "answer": TEXTO_ABSTENCION,
            "fragments": fragmentos or [],
            "abstained": True
        }


# Alias para mantener compatibilidad
RAGAdvancedPipeline = PipelineRAGAvanzado
