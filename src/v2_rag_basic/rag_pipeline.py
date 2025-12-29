"""
Pipeline de RAG Basico (Version 2).
Recupera fragmentos relevantes y genera respuestas usando el LLM con contexto.
"""

import faiss
import pickle
from pathlib import Path
from src.common.retriever.retriever import Recuperador
from src.v2_rag_basic.prompt import (
    construir_contexto_literal,
    construir_prompt_resumen_parcial,
    MAX_CARACTERES_CONTEXTO
)


class PipelineRAGBasico:
    """
    Pipeline de RAG basico que recupera fragmentos y genera respuestas.
    """
    
    def __init__(self, modelo_llm, directorio_base_datos="data", top_k=10, longitud_maxima_fragmento=400, indice_faiss=None, textos=None):
        """
        Inicializa el pipeline RAG basico.
        
        Args:
            modelo_llm: Instancia del modelo LLM
            directorio_base_datos: Carpeta donde estan los indices y textos
            top_k: Cuantos fragmentos recuperar
            longitud_maxima_fragmento: Limite de caracteres por fragmento
            indice_faiss: Indice FAISS precargado (opcional)
            textos: Lista de textos correspondiente al indice (opcional)
        """
        self.modelo_llm = modelo_llm
        self.top_k = top_k
        self.longitud_maxima_fragmento = longitud_maxima_fragmento
        self.directorio_base_datos = Path(directorio_base_datos)

        # Cargar FAISS y textos
        self.indice = indice_faiss
        self.textos = textos or []
        if self.indice is None or not self.textos:
            self.indice, self.textos = self._cargar_o_crear_indice()

        # Inicializar recuperador
        self.recuperador = Recuperador(directorio_base_datos=self.directorio_base_datos)

    def _cargar_o_crear_indice(self):
        """
        Carga indice FAISS y textos desde disco si existen,
        o devuelve un indice vacio y lista vacia.
        
        Returns:
            Tupla (indice_faiss, lista_textos)
        """
        ruta_indice = self.directorio_base_datos / "indices/faiss/index.faiss"
        ruta_textos = self.directorio_base_datos / "indices/faiss/texts.pkl"

        if ruta_indice.exists() and ruta_textos.exists():
            indice = faiss.read_index(str(ruta_indice))
            with open(ruta_textos, "rb") as archivo:
                textos = pickle.load(archivo)
            print(f"✅ Indice FAISS y textos cargados: {len(textos)} entradas")
        else:
            indice = None
            textos = []
            print("⚠️ Indice FAISS o textos no encontrados, iniciando vacio")

        return indice, textos

    def responder(self, pregunta: str) -> dict:
        """
        Recupera fragmentos relevantes y genera respuesta usando LLM.
        
        Args:
            pregunta: Pregunta del usuario
        
        Returns:
            Diccionario con:
            - question: Pregunta original
            - answer: Respuesta generada
            - fragments: Lista de fragmentos usados
        """
        fragmentos = self.recuperador.recuperar(pregunta, k=self.top_k)

        if not fragmentos:
            return {
                "question": pregunta,
                "answer": "El contexto proporcionado no contiene suficiente informacion para responder la pregunta.",
                "fragments": []
            }

        # Truncar fragmentos si son muy largos
        for fragmento in fragmentos:
            fragmento["text"] = fragmento["text"][:self.longitud_maxima_fragmento]

        # Construir contexto literal y limitar longitud total
        contexto = construir_contexto_literal(fragmentos)
        contexto = contexto[:MAX_CARACTERES_CONTEXTO]

        # Generar prompt y obtener respuesta
        prompt = construir_prompt_resumen_parcial(contexto, pregunta)
        respuesta = self.modelo_llm.generar(prompt).strip()

        return {
            "question": pregunta,
            "answer": respuesta,
            "fragments": fragmentos
        }


# Alias para mantener compatibilidad con codigo existente
RAGPipeline = PipelineRAGBasico
