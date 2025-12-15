import streamlit as st
from pathlib import Path
import sys

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar las tres versiones del sistema
from UI.run_baseline_ui import run_baseline_ui
from UI.run_rag_basic_ui import run_rag_basic_ui
from UI.run_rag_advanced_ui import run_rag_advanced_ui
from UI.metadata_init import inicializar_metadata_pdf
from UI.extraccion import preprocesar

# =============================
# Rutas base
# =============================
BASE_DATA = Path("UI/data")
PDF_DIR = BASE_DATA / "pdfs"
METADATA_CSV = BASE_DATA / "pdf_metadata.csv"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# Configuración de versiones
# =============================
VERSIONS = {
    "v1_baseline": {
        "name": "V1 - Baseline",
        "description": "Sin recuperación de información",
        "model": "Flan-T5 Base",
        "features": [
            "✓ Respuestas basadas en conocimiento interno",
            "✗ Sin acceso a documentos",
            "✗ Sin citación de fuentes"
        ],
        "function": run_baseline_ui,
        "icon": "🔵"
    },
    "v2_rag_basic": {
        "name": "V2 - RAG Básico",
        "description": "Recuperación simple con FAISS",
        "model": "Flan-T5 Base + FAISS",
        "features": [
            "✓ Recuperación de fragmentos relevantes",
            "✓ Acceso al contenido del documento",
            "✓ Respuestas contextualizadas"
        ],
        "function": run_rag_basic_ui,
        "icon": "🟢"
    },
    "v3_rag_advanced": {
        "name": "V3 - RAG Avanzado",
        "description": "Recuperación con citación y verificación",
        "model": "Flan-T5 Base + FAISS + Citations",
        "features": [
            "✓ Recuperación avanzada",
            "✓ Citación de fuentes (página + sección)",
            "✓ Verificación de evidencia",
            "✓ Abstención ante preguntas imposibles"
        ],
        "function": run_rag_advanced_ui,
        "icon": "🟣"
    }
}

# =============================
# Estilos CSS personalizados
# =============================
st.markdown("""
<style>
    /* Estilo general */
    .main {
        background-color: #f5f5f5;
    }
    
    /* Título principal */
    .main-title {
        text-align: center;
        color: #1f1f1f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Mensajes del chat */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .bot-message {
        background-color: #ffffff;
        border-left: 4px solid #4caf50;
    }
    
    .message-header {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .message-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #1f1f1f;
    }
    
    /* Iconos */
    .user-icon {
        color: #2196f3;
        font-size: 1.2rem;
    }
    
    .bot-icon {
        color: #4caf50;
        font-size: 1.2rem;
    }
    
    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 12px 20px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #2196f3;
        box-shadow: 0 0 0 2px rgba(33, 150, 243, 0.1);
    }
    
    /* Botones */
    .stButton > button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        background: linear-gradient(90deg, #2196f3 0%, #1976d2 100%);
        color: white;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4);
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 8px;
        border: 2px dashed #2196f3;
    }
    
    /* Version badge */
    .version-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .badge-v1 { background-color: #e3f2fd; color: #1976d2; }
    .badge-v2 { background-color: #e8f5e9; color: #388e3c; }
    .badge-v3 { background-color: #f3e5f5; color: #7b1fa2; }
</style>
""", unsafe_allow_html=True)

# =============================
# Configuración de página
# =============================
st.set_page_config(
    page_title="IntelliDocU - Document Q&A",
    page_icon="📚",
    layout="centered"
)

# =============================
# Inicializar estado de sesión
# =============================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

if 'selected_version' not in st.session_state:
    st.session_state.selected_version = "v1_baseline"

# =============================
# Funciones auxiliares
# =============================
def display_message(role, message, version_icon="🤖"):
    """Muestra un mensaje en el chat con estilos apropiados."""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">
                <span class="user-icon">👤</span>
                <span>Tú</span>
            </div>
            <div class="message-text">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">
                <span class="bot-icon">{version_icon}</span>
                <span>IntelliDocU</span>
            </div>
            <div class="message-text">{message}</div>
        </div>
        """, unsafe_allow_html=True)

def process_pdf(uploaded_file):
    """Procesa el PDF subido."""
    pdf_path = PDF_DIR / uploaded_file.name
    
    # Guardar PDF
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Procesar
    inicializar_metadata_pdf(pdf_path, METADATA_CSV)
    preprocesar(pdf_path)
    
    return pdf_path

# =============================
# Header
# =============================
st.markdown('<h1 class="main-title">📚 IntelliDocU</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema inteligente de preguntas y respuestas sobre documentos académicos</p>', unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
with st.sidebar:
    # Selector de versión
    st.header("🔧 Configuración")
    
    selected_version = st.selectbox(
        "Selecciona la versión del sistema",
        options=list(VERSIONS.keys()),
        format_func=lambda x: f"{VERSIONS[x]['icon']} {VERSIONS[x]['name']}",
        index=list(VERSIONS.keys()).index(st.session_state.selected_version)
    )
    
    # Si cambió la versión, limpiar chat
    if selected_version != st.session_state.selected_version:
        st.session_state.selected_version = selected_version
        st.session_state.chat_history = []
        st.rerun()
    
    # Información de la versión seleccionada
    version_info = VERSIONS[selected_version]
    st.markdown(f"**{version_info['description']}**")
    st.markdown(f"**Modelo:** {version_info['model']}")
    
    with st.expander("ℹ️ Características"):
        for feature in version_info['features']:
            st.markdown(f"- {feature}")
    
    st.markdown("---")
    
    # Subir PDF
    st.header("📄 Cargar Documento")
    
    uploaded_file = st.file_uploader(
        "Sube un archivo PDF",
        type=["pdf"],
        help="Selecciona un documento académico en formato PDF"
    )
    
    if uploaded_file is not None:
        # Verificar si es un nuevo PDF
        if st.session_state.current_pdf != uploaded_file.name:
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []  # Limpiar chat anterior
        
        if not st.session_state.pdf_processed:
            with st.spinner("🔄 Procesando documento..."):
                try:
                    pdf_path = process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.success("✅ Documento procesado correctamente")
                    st.info(f"📁 **Archivo:** {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error al procesar: {str(e)}")
        else:
            st.success("✅ Documento listo")
            st.info(f"📁 **Archivo:** {uploaded_file.name}")
    
    st.markdown("---")
    
    # Botón para limpiar chat
    if st.session_state.chat_history:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.chat_history = []
            st.rerun()

# =============================
# Área principal: Chat
# =============================

# Obtener la versión actual
current_version = VERSIONS[st.session_state.selected_version]

if not st.session_state.pdf_processed:
    # Estado inicial: sin PDF
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: white; border-radius: 12px; margin: 2rem 0;">
        <h3>👋 ¡Bienvenido a IntelliDocU!</h3>
        <p style="color: #666; font-size: 1.1rem; margin-top: 1rem;">
            Para comenzar, sube un documento PDF académico usando el panel lateral.
        </p>
        <p style="color: #999; margin-top: 1rem;">
            Una vez procesado, podrás hacer preguntas sobre su contenido.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Mostrar badge de versión activa
    badge_class = f"badge-{st.session_state.selected_version.split('_')[0]}"
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1rem;">
        <span class="version-badge {badge_class}">
            {current_version['icon']} {current_version['name']} activo
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # PDF procesado: mostrar chat
    st.markdown("### 💬 Conversación")
    
    # Contenedor del chat
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Haz tu primera pregunta sobre el documento para comenzar.")
        else:
            # Mostrar historial de chat
            for question, answer in st.session_state.chat_history:
                display_message("user", question)
                if answer:
                    display_message("bot", answer, current_version['icon'])
    
    # Input de pregunta (siempre al final)
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question_input = st.text_input(
            "Escribe tu pregunta aquí...",
            key="question_input",
            placeholder="Ej: ¿Cuál es el objetivo principal del documento?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("➤ Enviar", use_container_width=True)
    
    # Procesar pregunta
    if send_button and question_input:
        # Añadir pregunta al historial
        st.session_state.chat_history.append((question_input, None))
        
        # Generar respuesta usando la función de la versión seleccionada
        with st.spinner("🤔 Generando respuesta..."):
            try:
                answer = current_version['function'](question_input)
                # Actualizar con la respuesta
                st.session_state.chat_history[-1] = (question_input, answer)
            except Exception as e:
                answer = f"❌ Error al generar respuesta: {str(e)}"
                st.session_state.chat_history[-1] = (question_input, answer)
        
        # Recargar para mostrar la respuesta
        st.rerun()

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; padding: 1rem;">
    IntelliDocU - Sistema de Preguntas y Respuestas sobre Documentos Académicos
</div>
""", unsafe_allow_html=True)