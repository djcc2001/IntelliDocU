# 📚 IntelliDocU

<div align="center">

**Sistema experimental de Preguntas y Respuestas sobre Documentos Académicos usando RAG**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

*Reducción de alucinaciones en modelos generativos mediante técnicas de RAG*

</div>

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Características](#-características)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Versiones del Sistema](#-versiones-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Metodología](#-metodología)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Contribuciones](#-contribuciones)
- [Licencia](#-licencia)

---

## 🎯 Descripción

**IntelliDocU** es un sistema experimental diseñado para responder preguntas sobre documentos académicos en formato PDF utilizando modelos de lenguaje locales y técnicas de **Recuperación Aumentada con Generación (RAG)**. 

Este proyecto forma parte de un trabajo universitario orientado a analizar y reducir alucinaciones en modelos generativos, comparando tres versiones progresivas del sistema que incorporan mecanismos cada vez más sofisticados para minimizar respuestas incorrectas o sin evidencia.

### 🎓 Objetivo Principal

Desarrollar y evaluar un sistema tipo *ChatPDF* que:
- ✅ Responda preguntas precisas basadas en documentos académicos
- ✅ Cite las fuentes de información utilizadas
- ✅ Se abstenga cuando no hay evidencia suficiente
- ✅ Reduzca significativamente las alucinaciones

---

## ✨ Características

### 🔵 Versión 1 - Baseline
- Respuestas basadas únicamente en conocimiento interno del modelo
- Sin acceso a documentos
- Sin verificación de evidencia
- Línea base para comparación

### 🟢 Versión 2 - RAG Básico
- ✅ Recuperación semántica de fragmentos relevantes usando FAISS
- ✅ Respuestas contextualizadas con evidencia documental
- ✅ Grounding en el contenido del documento
- ⚠️ Sin citación obligatoria ni verificación cruzada

### 🟣 Versión 3 - RAG Avanzado
- ✅ Recuperación avanzada con umbrales ajustados
- ✅ **Citación explícita** de fuentes (documento, páginas, sección)
- ✅ **Verificación de evidencia** antes de responder
- ✅ **Abstención inteligente** cuando no hay información suficiente
- ✅ Control multinivel para reducir alucinaciones

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                      Interfaz de Usuario (Streamlit)          │
│                    UI/app.py - Selección de Versión           │
└───────────────────────┬───────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
    ┌───▼───┐      ┌───▼───┐      ┌───▼──────┐
    │  V1   │      │  V2   │      │    V3    │
    │Base-  │      │ RAG   │      │   RAG    │
    │line   │      │Basic  │      │ Advanced │
    └───┬───┘      └───┬───┘      └────┬─────┘
        │              │                │
        │         ┌────▼────┐      ┌────▼────┐
        │         │Retriever│      │Retriever│
        │         │ (FAISS) │      │ (FAISS) │
        │         └────┬────┘      └────┬────┘
        │              │                │
    ┌───▼──────────────▼────────────────▼───┐
    │      Modelo de Lenguaje (Qwen)        │
    │    Qwen/Qwen2.5-1.5B-Instruct         │
    └────────────────────────────────────────┘
```

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.8 o superior
- Git
- 8GB+ RAM recomendado (para cargar modelos)
- GPU opcional pero recomendada (CUDA)

### Pasos de Instalación

1. **Clonar el repositorio**
   ```bash
   git clone https://github.com/djcc2001/IntelliDocU.git
   cd IntelliDocU
   ```

2. **Crear entorno virtual**
   
   **Windows:**
   ```bash
   python -m venv env
   env\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar instalación**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
   ```

---

## 🎮 Uso Rápido

### Interfaz Web (Recomendado)

La forma más fácil de usar IntelliDocU es a través de la interfaz web:

```bash
streamlit run UI/app.py
```

Luego abre tu navegador en `http://localhost:8501`

**Pasos:**
1. 📄 Sube un PDF académico desde el panel lateral
2. ⏳ Espera a que se procese (extracción, limpieza, indexación)
3. 🔍 Selecciona la versión del sistema (V1, V2 o V3)
4. 💬 Haz preguntas sobre el documento
5. 📚 Obtén respuestas con citaciones (V3)

### Uso desde Línea de Comandos

#### Procesar un PDF
```bash
# El procesamiento se hace automáticamente al subir en la UI
# O manualmente:
python -m src.common.extract.extractor
python -m src.common.extract.cleaner
python -m src.common.chunking.chunker
python -m src.common.embeddings.build_faiss
```

#### Ejecutar Baseline (V1)
```bash
python -m src.v1_baseline.run_baseline
```

#### Ejecutar RAG Básico (V2)
```bash
python -m src.v2_rag_basic.run_rag
```

#### Ejecutar RAG Avanzado (V3)
```bash
python -m src.v3_rag_advanced.run_rag
```

#### Evaluación Completa
```bash
# Evaluar todas las versiones
python -m src.v1_baseline.run_baseline_eval
python -m src.v2_rag_basic.run_rag_eval
python -m src.v3_rag_advanced.run_rag_eval
```

---

## 🔬 Versiones del Sistema

### V1 - Baseline 🔵
**Sin recuperación de información**

- Modelo: Qwen/Qwen2.5-1.5B-Instruct
- Características:
  - Respuestas basadas en conocimiento interno
  - Sin acceso a documentos
  - Sin citación de fuentes
  - Puede generar alucinaciones

**Uso:** Línea base para comparación

### V2 - RAG Básico 🟢
**Recuperación simple con FAISS**

- Modelo: Qwen/Qwen2.5-1.5B-Instruct + FAISS
- Características:
  - ✅ Recuperación de fragmentos relevantes
  - ✅ Acceso al contenido del documento
  - ✅ Respuestas contextualizadas
  - ⚠️ Sin verificación de evidencia

**Uso:** Respuestas rápidas con contexto básico

### V3 - RAG Avanzado 🟣
**Recuperación con citación y verificación**

- Modelo: Qwen/Qwen2.5-1.5B-Instruct + FAISS + Verificación
- Características:
  - ✅ Recuperación avanzada con umbrales ajustados
  - ✅ **Citación de fuentes** (página + sección)
  - ✅ **Verificación de evidencia** antes de responder
  - ✅ **Abstención inteligente** ante preguntas imposibles
  - ✅ Control multinivel para reducir alucinaciones

**Uso:** Respuestas confiables y verificables (recomendado)

---

## 📁 Estructura del Proyecto

```
IntelliDocU/
│
├── 📄 README.md                 # Este archivo
├── 📋 requirements.txt          # Dependencias del proyecto
│
├── 📂 data/                      # Datos del proyecto
│   ├── pdfs/                     # PDFs académicos
│   ├── extracted/                # Texto extraído (JSONL)
│   ├── preprocessed/             # Texto limpio y normalizado
│   ├── fragments/                # Fragmentos para RAG
│   ├── indices/                  # Índices FAISS
│   │   └── faiss/
│   │       ├── index.faiss       # Índice vectorial
│   │       └── mapping.json      # Mapeo de fragmentos
│   ├── questions/                # Dataset de preguntas
│   └── pdf_metadata.csv          # Metadatos de PDFs
│
├── 📂 src/                       # Código fuente
│   ├── common/                   # Componentes compartidos
│   │   ├── extract/              # Extracción y limpieza
│   │   ├── chunking/             # Fragmentación de texto
│   │   ├── embeddings/           # Generación de embeddings
│   │   ├── retriever/            # Recuperación semántica
│   │   ├── llm/                  # Modelos de lenguaje
│   │   └── evaluation/           # Métricas y evaluación
│   │
│   ├── v1_baseline/              # Versión 1: Baseline
│   ├── v2_rag_basic/             # Versión 2: RAG Básico
│   └── v3_rag_advanced/         # Versión 3: RAG Avanzado
│
├── 📂 UI/                        # Interfaz de usuario
│   ├── app.py                    # Aplicación principal Streamlit
│   ├── run_baseline_ui.py        # Entrypoint V1
│   ├── run_rag_basic_ui.py      # Entrypoint V2
│   ├── run_rag_advanced_ui.py   # Entrypoint V3
│   ├── extraccion.py             # Pipeline de preprocesamiento
│   └── data/                     # Datos de la UI (replica)
│
├── 📂 results/                   # Resultados de evaluación
│   ├── v1_baseline/
│   ├── v2_rag_basic/
│   └── v3_rag_advanced/
│
└── 📂 docs/                      # Documentación del proyecto
    └── Fase_*.txt                # Documentación de fases
```

---

## 🧪 Metodología

El proyecto se desarrolló en 10 fases progresivas:

| Fase | Descripción | Archivos |
|------|-------------|----------|
| **0** | Preparación del entorno | `docs/Fase_0_preparacion_entorno.txt` |
| **1** | Metadatos del dataset | `docs/Fase_1_metadatos_del_dataset.txt` |
| **2** | Extracción y limpieza | `docs/Fase_2_extraccion_y_limpieza_del_texto.txt` |
| **3** | Chunking y preparación | `docs/Fase_3_chunking_y_preparacion_para_RAG.txt` |
| **4** | Embeddings e índice FAISS | `docs/Fase_4_embeddings_y_construccion_del_indice_vectorial.txt` |
| **5** | Recuperación semántica | `docs/Fase_5_recuperacion_semantica_con_FAISS.txt` |
| **6** | V1 Baseline | `docs/Fase_6_v1_baseline.txt` |
| **7** | V2 RAG Básico | `docs/Fase_7_v2_RAG_Basic_Recuperacion_y_Grounding.txt` |
| **8** | V3 RAG Avanzado | `docs/Fase_8_v3_RAG_avanzado.txt` |
| **9** | Métricas y evaluación | `docs/Fase_9_metricas.txt` |
| **10** | Interfaz de usuario | `docs/Fase_10_interfaz.txt` |

Cada fase está documentada en detalle en los archivos `docs/Fase_*.txt` correspondientes.

---

## 🛠️ Tecnologías Utilizadas

### Modelos de Lenguaje
- **Qwen/Qwen2.5-1.5B-Instruct** - Modelo principal para generación
- **sentence-transformers/all-MiniLM-L6-v2** - Modelo de embeddings

### Librerías Principales
- **PyTorch** - Framework de deep learning
- **Transformers** - Modelos pre-entrenados de HuggingFace
- **FAISS** - Búsqueda vectorial eficiente
- **Streamlit** - Interfaz web interactiva
- **PyMuPDF** - Extracción de texto de PDFs
- **Pandas** - Manejo de datos estructurados

### Herramientas
- **Python 3.8+** - Lenguaje de programación
- **Git** - Control de versiones
- **CUDA** (opcional) - Aceleración GPU

---

## 📊 Ejemplo de Uso

### Ejemplo 1: Pregunta Factual

**Pregunta:** "¿Cuál es el objetivo principal del documento?"

**V1 Baseline:**
```
El objetivo principal de un documento académico típicamente es...
[Respuesta genérica sin acceso al documento]
```

**V2 RAG Básico:**
```
Según el documento, el objetivo principal es analizar...
[Respuesta basada en fragmentos recuperados]
```

**V3 RAG Avanzado:**
```
El objetivo principal del documento es desarrollar un sistema...
📚 Evidencia: [Doc: arxiv_251210894_duetsvg, Paginas: 1, 2, Sec: abstract]
```

### Ejemplo 2: Pregunta Imposible

**Pregunta:** "¿Qué dice el documento sobre la teoría de la relatividad?"

**V1 Baseline:**
```
La teoría de la relatividad establece que...
[Alucinación - responde aunque no esté en el documento]
```

**V2 RAG Básico:**
```
El documento no menciona específicamente la teoría de la relatividad...
[Respuesta parcialmente correcta]
```

**V3 RAG Avanzado:**
```
No se menciona en el documento.
[Abstención correcta - no hay evidencia]
```

---

## 🤝 Contribuciones

Este es un proyecto académico. Las contribuciones son bienvenidas:

1. 🍴 Haz un Fork del proyecto
2. 🌿 Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. 📤 Push a la rama (`git push origin feature/AmazingFeature`)
5. 🔄 Abre un Pull Request

### Áreas de Mejora
- ⚡ Optimización de rendimiento
- 🧪 Nuevas métricas de evaluación
- 📝 Mejora de documentación
- 🐛 Corrección de bugs
- ✨ Nuevas características

---

## 📜 Licencia

Este proyecto es de **uso académico**. No se permite uso comercial sin autorización del autor.

<!-----

## 👥 Autores

- **Deni** - *Desarrollo inicial* - [TuGitHub](https://github.com/tu-usuario)-->

---

<div align="center">

**⭐ Si este proyecto te resultó útil, considera darle una estrella ⭐**

</div>
