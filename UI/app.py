import streamlit as st
from pathlib import Path
import sys
import time

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
    /* Ocultar header (barra superior de Streamlit) */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Ocultar footer */
    footer {
        display: none;
    }

    /* Quitar padding superior extra que deja Streamlit */
    .block-container {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

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
    
    /* Contenedor de conversación */
    .conversation-container {
        background-color: #161b22;
        border-radius: 12px;
        padding: 2rem;

        height: 400px;              /* altura fija */
        overflow-y: auto;           /* scroll vertical */
        overflow-x: hidden;         /* no scroll horizontal */

        margin-bottom: 1rem;
    }

    
    /* Mensajes del chat - Estilo simplificado */
    .chat-line {
        margin-bottom: 1.5rem;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    .chat-line strong {
        font-weight: 600;
    }
    
    .user-line {
        color: #ffffff;
    }
    
    .bot-line {
        color: #f6ff52;
    }
    
    .loading-line {
        color: #f6ff52;
        font-style: italic;
    }
    
    /* Animación de puntos suspensivos */
    .loading-dots::after {
        content: '...';
        animation: dots 1.5s steps(4, end) infinite;
    }
    
    @keyframes dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    /* Mensaje inicial */
    .initial-message {
        text-align: center;
        color: #999;
        padding: 3rem 1rem;
        font-size: 1rem;
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
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;          
        line-height: 1;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.4);
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
    
    /* Tooltip personalizado */
    .version-selector {
        position: relative;
        display: inline-block;
    }
    
    .tooltip-content {
        visibility: hidden;
        background-color: #333;
        color: #fff;
        text-align: left;
        border-radius: 8px;
        padding: 12px;
        position: absolute;
        z-index: 1000;
        left: 100%;
        margin-left: 10px;
        top: 0;
        width: 250px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .version-selector:hover .tooltip-content {
        visibility: visible;
        opacity: 1;
    }
    
    .tooltip-content h4 {
        margin: 0 0 8px 0;
        font-size: 0.9rem;
        color: #fff;
    }
    
    .tooltip-content ul {
        margin: 0;
        padding-left: 16px;
        list-style: none;
    }
    
    .tooltip-content li {
        margin: 4px 0;
        font-size: 0.8rem;
    }
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

if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

# =============================
# Funciones auxiliares
# =============================
def escape_html(text):
    """Escapa caracteres HTML para prevenir problemas de renderizado."""
    import html
    return html.escape(text)

def display_conversation():
    """Muestra toda la conversación en formato simplificado."""
    if not st.session_state.chat_history and not st.session_state.is_loading:
        st.markdown('''
            <div class="conversation-container">
            </div>
        ''', unsafe_allow_html=True)
    else:
        conversation_html = '<div class="conversation-container">'
        
        for question, answer in st.session_state.chat_history:
            # Pregunta del usuario (escapar HTML)
            safe_question = escape_html(question)
            conversation_html += f'<div class="chat-line user-line"><strong>User:</strong> {safe_question}</div>'
            
            # Respuesta del bot
            if answer:
                safe_answer = escape_html(answer)
                conversation_html += f'<div class="chat-line bot-line"><strong>IntelliDocU:</strong> {safe_answer}</div>'
        
        # Si está cargando, mostrar puntos suspensivos
        if st.session_state.is_loading:
            conversation_html += '<div class="chat-line loading-line"><strong>IntelliDocU:</strong> <span class="loading-dots"></span></div>'
        
        conversation_html += '</div>'
        st.markdown(conversation_html, unsafe_allow_html=True)

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
#st.markdown('<h1 class="main-title">📚 IntelliDocU</h1>', unsafe_allow_html=True)
#st.markdown('<p class="subtitle">Sistema inteligente de preguntas y respuestas sobre documentos académicos</p>', unsafe_allow_html=True)

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("🔧 Configuración")
    
    # Selector de versión con tooltip
    st.markdown("**Versión del Sistema**")
    
    selected_version = st.selectbox(
        "modelo",
        options=list(VERSIONS.keys()),
        format_func=lambda x: f"{VERSIONS[x]['icon']} {VERSIONS[x]['name']}",
        index=list(VERSIONS.keys()).index(st.session_state.selected_version),
        label_visibility="collapsed",
        help="Selecciona el modelo de IA a utilizar. Pasa el cursor sobre cada opción para ver sus características."
    )
    
    # Si cambió la versión, limpiar chat
    if selected_version != st.session_state.selected_version:
        st.session_state.selected_version = selected_version
        st.session_state.chat_history = []
        st.session_state.is_loading = False
        st.rerun()
    
    # Información compacta de la versión
    version_info = VERSIONS[selected_version]
    st.markdown(f"<small><i>{version_info['description']}</i></small>", unsafe_allow_html=True)
    
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
            st.session_state.chat_history = []
            st.session_state.is_loading = False
        
        if not st.session_state.pdf_processed:
            with st.spinner("🔄 Procesando documento..."):
                try:
                    pdf_path = process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.success("✅ Documento procesado")
                    st.info(f"📁 {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.success("✅ Documento listo")
            st.info(f"📁 {uploaded_file.name}")
    
    st.markdown("---")
    
    # Botón para limpiar chat
    if st.session_state.chat_history:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.chat_history = []
            st.session_state.is_loading = False
            st.rerun()

# =============================
# Área principal
# =============================

current_version = VERSIONS[st.session_state.selected_version]

if not st.session_state.pdf_processed:
    # Estado inicial: sin PDF
    st.markdown("""
    <div style="text-align: center; padding: 3rem; background-color: #161b22; border-radius: 12px; margin: 2rem 0;">
        <h3>👋 ¡Bienvenido a IntelliDocU!</h3>
        <p style="color: white; font-size: 1.1rem; margin-top: 1rem;">
            Para comenzar, sube un documento PDF académico usando el panel lateral.
        </p>
        <p style="color: #999; margin-top: 1rem;">
            Una vez procesado, podrás hacer preguntas sobre su contenido.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    # Badge de versión
    badge_class = f"badge-{st.session_state.selected_version.split('_')[0]}"
    
    # Mostrar conversación
    display_conversation()
    
    # Input de pregunta
    st.markdown("---")
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        question_input = st.text_input(
            "Escribe tu pregunta aquí...",
            key="question_input",
            placeholder="Ej: ¿Cuál es el objetivo principal del documento?",
            label_visibility="collapsed",
            disabled=st.session_state.is_loading
        )
    
    with col2:
        send_button = st.button("➤ Enviar", use_container_width=True, disabled=st.session_state.is_loading)
    
    # Procesar pregunta
    if send_button and question_input:
        # Añadir pregunta al historial
        st.session_state.chat_history.append((question_input, None))
        st.session_state.is_loading = True
        
        # Recargar para mostrar el estado de carga
        st.rerun()
    
    # Si está cargando, generar respuesta
    if st.session_state.is_loading:
        try:
            answer = current_version['function'](st.session_state.chat_history[-1][0])
            # Actualizar con la respuesta
            st.session_state.chat_history[-1] = (st.session_state.chat_history[-1][0], answer)
        except Exception as e:
            answer = f"❌ Error al generar respuesta: {str(e)}"
            st.session_state.chat_history[-1] = (st.session_state.chat_history[-1][0], answer)
        
        st.session_state.is_loading = False
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