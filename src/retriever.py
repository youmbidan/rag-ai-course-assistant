import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DB_PATH, EMBEDDING_MODEL, TOP_K


@st.cache_resource(show_spinner=False)
def get_vector_store():
    """Charge le modèle d'embedding et la base vectorielle une seule fois, mise en cache par Streamlit."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_PATH, embedding_function=embeddings)


def retrieve_chunks(question, k=TOP_K):
    """Retourne les k chunks les plus proches de la question.

    Note : on a testé un filtrage par seuil de distance pour détecter les questions
    hors-sujet, mais les scores se chevauchaient trop entre questions pertinentes et
    non pertinentes sur ce corpus (avec ce modèle d'embedding) pour fixer un seuil fiable.
    La détection du hors-sujet est donc déléguée au LLM via le prompt (voir rag_chain.py),
    qui juge mieux la pertinence sémantique fine que la seule distance cosinus.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(question, k=k)