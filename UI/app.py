# UI/app.py
import streamlit as st
from pathlib import Path
import sys
import time
import html

# Agregar raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Importar las tres versiones del sistema
from UI.run_baseline_ui import run_baseline_ui
from UI.run_rag_basic_ui import run_rag_basic_ui
from UI.run_rag_advanced_ui import run_rag_advanced_ui, init_retriever
from UI.metadata_init import inicializar_metadata_pdf
from UI.extraccion import preprocesar

# =============================
# Rutas base
# =============================
BASE_DATA = PROJECT_ROOT / "UI/data"
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
st.markdown("""<style>
/* Todo tu CSS de app.py original */
header[data-testid="stHeader"] {display: none;}
footer {display: none;}
.block-container {padding-top: 1rem;}
.conversation-container {background-color:#161b22;border-radius:12px;padding:2rem;height:400px;overflow-y:auto;overflow-x:hidden;margin-bottom:1rem;}
.chat-line {margin-bottom:1.5rem;font-size:1rem;line-height:1.8;}
.chat-line strong {font-weight:600;}
.user-line {color:#fff;}
.bot-line {color:#f6ff52;}
.loading-line {color:#f6ff52;font-style:italic;}
.loading-dots::after {content:'...';animation:dots 1.5s steps(4,end) infinite;}
@keyframes dots {0%,20% {content:'.';} 40% {content:'..';} 60%,100% {content:'...';}}
.stTextInput > div > div > input {border-radius:20px;padding:12px 20px;border:2px solid #e0e0e0;font-size:1rem;}
.stTextInput > div > div > input:focus {border-color:#2196f3;box-shadow:0 0 0 2px rgba(33,150,243,0.1);}
.stButton > button {display:inline-flex;align-items:center;justify-content:center;gap:0.4rem;line-height:1;}
.stButton > button:hover {transform:translateY(-2px);box-shadow:0 4px 12px rgba(33,150,243,0.4);}
.badge-v1 { background-color: #e3f2fd; color: #1976d2; }
.badge-v2 { background-color: #e8f5e9; color: #388e3c; }
.badge-v3 { background-color: #f3e5f5; color: #7b1fa2; }
</style>""", unsafe_allow_html=True)

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
if 'rag_advanced_retriever' not in st.session_state:
    st.session_state.rag_advanced_retriever = init_retriever(base_data_dir=str(BASE_DATA))

# =============================
# Funciones auxiliares
# =============================
def escape_html(text):
    return html.escape(text)

def display_conversation():
    if not st.session_state.chat_history and not st.session_state.is_loading:
        st.markdown('<div class="conversation-container"></div>', unsafe_allow_html=True)
    else:
        conversation_html = '<div class="conversation-container">'
        for question, answer in st.session_state.chat_history:
            conversation_html += f'<div class="chat-line user-line"><strong>User:</strong> {escape_html(question)}</div>'
            if answer:
                conversation_html += f'<div class="chat-line bot-line"><strong>IntelliDocU:</strong> {escape_html(answer)}</div>'
        if st.session_state.is_loading:
            conversation_html += '<div class="chat-line loading-line"><strong>IntelliDocU:</strong> <span class="loading-dots"></span></div>'
        conversation_html += '</div>'
        st.markdown(conversation_html, unsafe_allow_html=True)

def process_pdf(uploaded_file):
    pdf_path = PDF_DIR / uploaded_file.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    inicializar_metadata_pdf(pdf_path, METADATA_CSV)
    preprocesar(pdf_path)
    return pdf_path

# =============================
# Sidebar
# =============================
with st.sidebar:
    st.header("🔧 Configuración")
    selected_version = st.selectbox(
        "modelo",
        options=list(VERSIONS.keys()),
        format_func=lambda x: f"{VERSIONS[x]['icon']} {VERSIONS[x]['name']}",
        index=list(VERSIONS.keys()).index(st.session_state.selected_version),
        label_visibility="collapsed",
        help="Selecciona el modelo de IA a utilizar. Pasa el cursor sobre cada opción para ver sus características."
    )
    if selected_version != st.session_state.selected_version:
        st.session_state.selected_version = selected_version
        st.session_state.chat_history = []
        st.session_state.is_loading = False
        st.rerun()

    st.markdown("---")
    st.header("📄 Cargar Documento")
    uploaded_file = st.file_uploader("Sube un archivo PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.session_state.current_pdf != uploaded_file.name:
            st.session_state.current_pdf = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.chat_history = []
            st.session_state.is_loading = False
        if not st.session_state.pdf_processed:
            with st.spinner("🔄 Procesando documento..."):
                try:
                    process_pdf(uploaded_file)
                    st.session_state.pdf_processed = True
                    st.success("✅ Documento procesado")
                    st.info(f"📁 {uploaded_file.name}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.success("✅ Documento listo")
            st.info(f"📁 {uploaded_file.name}")

    # Botón limpiar chat
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
    # Badge
    badge_class = f"badge-{st.session_state.selected_version.split('_')[0]}"
    display_conversation()
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

    if send_button and question_input:
        st.session_state.chat_history.append((question_input, None))
        st.session_state.is_loading = True
        st.rerun()

    if st.session_state.is_loading:
        try:
            if st.session_state.selected_version == "v3_rag_advanced":
                answer = current_version['function'](
                    st.session_state.chat_history[-1][0],
                    retriever=st.session_state.rag_advanced_retriever
                )
            else:
                answer = current_version['function'](st.session_state.chat_history[-1][0])
            st.session_state.chat_history[-1] = (st.session_state.chat_history[-1][0], answer)
        except Exception as e:
            st.session_state.chat_history[-1] = (st.session_state.chat_history[-1][0], f"❌ Error: {str(e)}")
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
