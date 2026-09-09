import os
import streamlit as st
from src.rag_chain import answer_question
from src.ingest import load_pdfs, split_documents, build_vector_store
from src.config import PDF_FOLDER
from src.ingest import rebuild_index
from src.retriever import get_vector_store

st.set_page_config(page_title="Assistant Cours IA", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1c1210; }

    .greeting-star { text-align: center; font-size: 2.6rem; color: #b89a82; margin-top: 2.5rem; }
    .greeting-title { text-align: center; font-size: 2.2rem; font-weight: 800; color: #f2e9e1; margin-bottom: 0.3rem; }
    .greeting-subtitle { text-align: center; color: #c9b8ab; font-size: 1.15rem; margin-bottom: 2rem; }

    div[data-testid="stForm"] input {
        border-radius: 30px !important;
        border: 2px solid #7a4b3a !important;
        padding: 0.9rem 1.3rem !important;
        background-color: #2a1a15 !important;
        color: #f2e9e1 !important;
        font-size: 1.05rem !important;
    }
    div[data-testid="stForm"] button {
        border-radius: 30px !important;
        background-color: #b89a82 !important;
        color: #1c1210 !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        height: 100% !important;
    }
    div[data-testid="stForm"] button:hover { background-color: #d4bba3 !important; }

    .source-badge {
        display: inline-block; background: #7a4b3a; color: #ffffff;
        font-size: 0.85rem; font-weight: 700; padding: 4px 12px;
        border-radius: 6px; margin-right: 5px; margin-bottom: 5px;
    }

    [data-testid="stSidebar"] * { font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.markdown("### Assistant Cours IA")

    if st.button("+ Nouvelle discussion", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.text_input("Rechercher dans l'historique", placeholder="Rechercher dans l'historique...",
                  label_visibility="collapsed")

    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.caption("HISTORIQUE")

    past_questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if past_questions:
        for q in past_questions:
            st.markdown(f"- {q[:38]}{'...' if len(q) > 38 else ''}")
    else:
        st.caption("Aucune question pour l'instant")

    st.markdown("---")
    st.caption("MES COURS")

    with st.expander("Ajouter un cours (PDF)"):
        uploaded_files = st.file_uploader("Dépose tes PDF de cours", type=["pdf"], accept_multiple_files=True)
        if uploaded_files and st.button("Indexer ces cours", use_container_width=True):
            with st.spinner("Indexation en cours..."):
                for file in uploaded_files:
                    dest = PDF_FOLDER / file.name
                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())
                rebuild_index()
                get_vector_store.clear()  # vide le cache pour recharger la nouvelle base

            st.success(f"{len(uploaded_files)} cours ajouté(s) et indexé(s).")
            st.rerun()

    existing_pdfs = list(PDF_FOLDER.glob("*.pdf"))
    if existing_pdfs:
        for pdf in existing_pdfs:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.caption(pdf.name)
            with col2:
                if st.button("✕", key=f"del_{pdf.name}", help=f"Supprimer {pdf.name}"):
                    pdf.unlink()
                    with st.spinner("Mise à jour de l'index..."):
                        rebuild_index()
                        get_vector_store.clear()
                    st.rerun()
    else:
        st.caption("Aucun cours indexé pour l'instant.")

# --- Écran d'accueil ---
if not st.session_state.messages:
    st.markdown("<div class='greeting-star'>★</div>", unsafe_allow_html=True)
    st.markdown("<div class='greeting-title'>Bonjour</div>", unsafe_allow_html=True)
    st.markdown("<div class='greeting-subtitle'>Comment puis-je t'aider aujourd'hui ?</div>", unsafe_allow_html=True)

# --- Historique ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("chunks"):
            with st.expander("Sources consultées"):
                for chunk in message["chunks"]:
                    source = os.path.basename(chunk.metadata.get("source", "Document"))
                    page = chunk.metadata.get("page", "-")
                    st.markdown(f"<span class='source-badge'>{source} — page {page}</span>", unsafe_allow_html=True)
                    st.caption(f'"{chunk.page_content[:200]}..."')

# --- Champ de saisie ---
with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([8, 1])
    with col1:
        prompt = st.text_input(
            "Message", placeholder="Écris ton message ici...", label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("➤", use_container_width=True)

if submitted and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Analyse des documents..."):
        answer, chunks = answer_question(prompt, history=st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": answer, "chunks": chunks})
    st.rerun()