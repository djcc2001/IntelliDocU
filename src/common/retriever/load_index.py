"""
Modulo para cargar el indice FAISS y sus metadatos desde disco.
Maneja casos donde el indice aun no existe (proyecto nuevo sin documentos).
"""

import faiss
import json
from pathlib import Path


def cargar_indice_faiss(directorio_base_datos="data"):
    """
    Carga el indice FAISS desde disco.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Returns:
        Indice FAISS o None si no existe o esta vacio
    """
    base = Path(directorio_base_datos)
    ruta_indice = base / "indices" / "faiss" / "index.faiss"

    # Verificar que el archivo existe
    if not ruta_indice.exists():
        return None

    try:
        # Verificar que el archivo no esta vacio
        if ruta_indice.stat().st_size == 0:
            return None
    except Exception:
        return None

    # Cargar indice FAISS
    return faiss.read_index(str(ruta_indice))


def cargar_mapeo(directorio_base_datos="data"):
    """
    Carga el mapeo de indices a metadatos de fragmentos.
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Returns:
        Lista de metadatos o lista vacia si no existe
    """
    base = Path(directorio_base_datos)
    ruta_mapeo = base / "indices" / "faiss" / "mapping.json"

    if not ruta_mapeo.exists():
        return []

    try:
        with open(ruta_mapeo, "r", encoding="utf-8") as archivo:
            datos = json.load(archivo)
            return datos if isinstance(datos, list) else []
    except Exception:
        return []


def cargar_metadatos_indice(directorio_base_datos="data"):
    """
    Carga los metadatos del indice (modelo usado, dimension, etc.).
    
    Args:
        directorio_base_datos: Directorio base donde estan los datos
    
    Returns:
        Diccionario con metadatos o diccionario vacio si no existe
    """
    base = Path(directorio_base_datos)
    ruta_metadatos = base / "indices" / "faiss" / "index_meta.json"

    if not ruta_metadatos.exists():
        return {}

    try:
        with open(ruta_metadatos, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception:
        return {}


# Funciones alias para mantener compatibilidad con codigo existente
load_faiss_index = cargar_indice_faiss
load_mapping = cargar_mapeo
load_index_meta = cargar_metadatos_indice
