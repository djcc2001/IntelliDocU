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
        # Umbrales reducidos para ser menos estricto
        if puntuacion >= 0.25:
            return "strong"
        if puntuacion >= 0.10:  # Reducido de 0.18 a 0.10
            return "weak"
        return "none"

    def respuesta_usa_contexto(self, respuesta, fragmentos):
        """
        Verifica si la respuesta usa palabras clave del contexto.
        Version mas flexible para ser menos estricto.
        
        Args:
            respuesta: Respuesta generada por el LLM
            fragmentos: Fragmentos usados como contexto
        
        Returns:
            True si la respuesta parece usar el contexto, False en caso contrario
        """
        if not fragmentos or not respuesta:
            return False
            
        respuesta_minusculas = respuesta.lower()
        texto_contexto = " ".join(fragmento["text"].lower() for fragmento in fragmentos)

        # Extraer palabras clave (palabras de 4+ caracteres, mas flexible)
        palabras_clave = set(re.findall(r"\b[a-zA-Z]{4,}\b", texto_contexto))
        
        # Verificar si hay al menos algunas palabras clave en la respuesta
        palabras_encontradas = sum(1 for palabra in palabras_clave if palabra in respuesta_minusculas)
        
        # Si hay al menos 2 palabras clave o si la respuesta es suficientemente larga, considerar que usa el contexto
        if palabras_encontradas >= 2:
            return True
        
        # Si la respuesta es muy corta y no tiene palabras clave, probablemente no usa el contexto
        if len(respuesta_minusculas.split()) < 5:
            return palabras_encontradas >= 1
        
        # Para respuestas mas largas, ser mas permisivo
        return palabras_encontradas >= 1 or len(respuesta_minusculas) > 50

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

        # Recuperar fragmentos relevantes (umbral reducido para ser menos estricto)
        fragmentos = self.recuperador.recuperar(
            pregunta,
            k=self.top_k,
            puntuacion_minima=0.10  # Reducido de 0.18 a 0.10
        )

        # Verificar fuerza de evidencia - permitir respuestas con evidencia "weak"
        fuerza = self.fuerza_evidencia(fragmentos)
        if fuerza == "none":
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
