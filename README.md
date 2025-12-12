# IntelliDocU
Sistema experimental para responder preguntas sobre documentos académicos en PDF utilizando modelos de lenguaje locales y técnicas de Recuperación Aumentada con Generación (RAG). El proyecto forma parte de un trabajo universitario orientado a analizar y reducir alucinaciones en modelos generativos.

---

## 📘 Objetivo del Proyecto
El propósito principal es desarrollar y evaluar un sistema tipo *ChatPDF*, comparando distintas versiones que incorporan cada vez más mecanismos para minimizar respuestas incorrectas o sin evidencia.

Las versiones incluyen:

- **v1 – Baseline:** Búsqueda simple sin RAG, sin verificación.
- **v2 – RAG Básico:** Recuperación de fragmentos + respuesta condicionada al contexto.
- **v3 – RAG Avanzado:** Verificación cruzada, abstención y chequeo de evidencia (NLI).

---

## 🏗️ Estructura del Proyecto
```bash
IntelliDocU/
    data/ # PDFs y dataset de preguntas-respuestas
    indices/ # Índices vectoriales generados por FAISS
    results/ # Métricas, logs y resultados de evaluación
    src/
        common/ # Funciones reutilizables (lectura PDF, chunking, etc.)
        v1_baseline/ # Implementación de la primera versión
        v2_rag_basic/ # Implementación del RAG simple
        v3_rag_advanced/ # Implementación con verificación y abstención
    requirements.txt
    README.md
```
---

## 🧪 Metodología Resumida

1. **Fase 0 — Preparación del entorno**
   - Creación del entorno virtual.
   - Instalación de dependencias mínimas.
   - Estructura base del proyecto.
   - Configuración de Git y GitHub.

2. **Fase 1 — Dataset**
   - Reunir PDFs académicos variados.
   - Crear conjuntos de preguntas:
     - 40% factuales
     - 30% de localización
     - 30% imposibles
   - Crear dataset en formato JSON/CSV.

3. **Fase 2 — Preprocesamiento y Embeddings**
   - Limpieza y segmentación (chunking) del texto.
   - Generación de vectores.
   - Creación del índice FAISS.

4. **Fase 3 — Implementación V1 (Baseline)**
   - Respuestas sin verificación ni recuperación avanzada.

5. **Fase 4 — Implementación V2 (RAG básico)**
   - Recuperación y uso de contexto.
   - Ajuste de prompts.

6. **Fase 5 — Implementación V3 (RAG avanzado)**
   - Verificación cruzada (NLI).
   - Detección de alucinaciones.
   - Abstención cuando no hay evidencia.

7. **Fase 6 — Evaluación**
   - Exactitud.
   - Verifiability.
   - Abstention accuracy.
   - Comparación entre versiones.

---

## 🚀 Cómo ejecutar el proyecto

### 1. Activar el entorno virtual

**Windows**
```bash
env\Scripts\activate
```

**Linux/Mac**
```bash
source env/bin/activate
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### Ejecutar una versión específica (ejemplo)
```bash
python src/v1_baseline/main.py
```

---

## 📂 Dataset
Los documentos PDF utilizados son almacenados en la carpeta:
```bash
data/
```
La estructura recomendada para preguntas:
```bash
{
  "pdf_id": "documento1",
  "question": "¿Cuál es el objetivo principal del texto?",
  "answer": "Objetivo X",
  "type": "factual",
  "page": 3
}
```

---

## 📊 Resultados
Los resultados de métricas y pruebas se guardan automáticamente en:
```bash
results/
```

---

## 🤝 Contribuciones
Este proyecto es académico y no busca producción comercial.
Puedes aportar con mejoras o sugerencias abriendo issues o pull requests.

---

## 📜 Licencia
Uso académico. No se permite uso comercial sin autorización del autor.

---