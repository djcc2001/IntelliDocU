"""
Modulo de recuperacion semantica usando FAISS.
Permite buscar fragmentos de texto relevantes para una pregunta usando similitud de embeddings.
"""

import numpy as np
from src.common.embeddings.embedder import GeneradorEmbeddings
from src.common.retriever.load_index import (
    cargar_indice_faiss,
    cargar_mapeo,
    cargar_metadatos_indice
)


class Recuperador:
    """
    Recuperador basado en FAISS (similitud coseno).
    Funciona incluso cuando aun no hay documentos indexados.
    """

    def __init__(
        self,
        directorio_base_datos="data",
        nombre_modelo="sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Inicializa el recuperador.
        
        Args:
            directorio_base_datos: Directorio base donde estan los indices y datos
            nombre_modelo: Nombre del modelo de embeddings a utilizar
        """
        self.directorio_base_datos = directorio_base_datos
        self.generador_embeddings = GeneradorEmbeddings(nombre_modelo)

        # Carga segura de artefactos
        self.indice = cargar_indice_faiss(directorio_base_datos)
        self.mapeo = cargar_mapeo(directorio_base_datos)
        self.metadatos_indice = cargar_metadatos_indice(directorio_base_datos)

        # Normalizar mapeo → siempre lista
        if not isinstance(self.mapeo, list):
            self.mapeo = []

        # Estado del indice
        if self.indice is None:
            self.similitud = None
            self.tipo_indice = "none"
            print("ℹRecuperador inicializado SIN indice FAISS (no hay documentos)")
        else:
            self.similitud = self.metadatos_indice.get("similarity", "cosine")
            self.tipo_indice = self._detectar_tipo_indice()
            print(f"Recuperador listo — indice: {self.tipo_indice}, sim: {self.similitud}")

        print(f"Recuperador usando directorio_base_datos = {self.directorio_base_datos}")

    def _detectar_tipo_indice(self):
        """
        Detecta el tipo de indice FAISS usado.
        
        Returns:
            String con el tipo de indice detectado
        """
        if self.indice is None:
            return "none"

        nombre = self.indice.__class__.__name__
        if "IP" in nombre:
            return "IP"  # Producto Interno (Inner Product)
        if "L2" in nombre:
            return "L2"  # Distancia Euclidiana
        return nombre

    def tiene_indice(self) -> bool:
        """
        Indica si el recuperador esta listo para buscar.
        
        Returns:
            True si hay indice y mapeo disponibles, False en caso contrario
        """
        return self.indice is not None and len(self.mapeo) > 0

    def recuperar(
        self,
        consulta: str,
        k: int = 5,
        secciones_permitidas=None,
        puntuacion_minima: float = 0.0
    ):
        """
        Recupera los k fragmentos mas relevantes para una consulta.
        
        Args:
            consulta: Texto de la pregunta o consulta
            k: Numero de fragmentos a recuperar
            secciones_permitidas: Lista de secciones permitidas (None = todas)
            puntuacion_minima: Puntuacion minima de similitud para considerar un fragmento
        
        Returns:
            Lista de fragmentos ordenados por relevancia (mayor a menor)
        """
        # No hay indice → no buscar
        if not self.tiene_indice():
            return []

        # Generar embedding de la consulta
        vector_consulta = self.generador_embeddings.codificar([consulta]).astype("float32")

        # Buscar mas resultados de los necesarios para filtrar por seccion
        buscar_k = max(k * 5, k)
        puntuaciones, indices = self.indice.search(vector_consulta, buscar_k)

        resultados = []
        resultados_respaldo = []

        for puntuacion, indice in zip(puntuaciones[0], indices[0]):
            # Validar indice
            if indice < 0 or indice >= len(self.mapeo):
                continue

            fragmento = dict(self.mapeo[indice])
            fragmento["score"] = float(puntuacion)

            # Filtrar por puntuacion minima
            if puntuacion < puntuacion_minima:
                continue

            seccion = fragmento.get("section", "unknown")

            # Filtrar por secciones permitidas
            if secciones_permitidas is None or seccion in secciones_permitidas:
                resultados.append(fragmento)
            else:
                resultados_respaldo.append(fragmento)

        # Ordenar por puntuacion (mayor = mejor)
        resultados.sort(key=lambda x: x["score"], reverse=True)
        resultados_respaldo.sort(key=lambda x: x["score"], reverse=True)

        # Completar con resultados de respaldo si faltan
        if len(resultados) < k:
            resultados.extend(resultados_respaldo[: k - len(resultados)])

        return resultados[:k]


# Alias para mantener compatibilidad con codigo existente
Retriever = Recuperador
