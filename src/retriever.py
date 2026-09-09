from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DB_PATH, EMBEDDING_MODEL, TOP_K

def load_vector_store():
    """Fonction qui recharge la base vectorielle déjà créée par ingest.py."""

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

def retrieve_chunks(question, k=TOP_K):
    "Fonction qui retrouve les k chunks les plus pertinents pour une question donnée."
    vector_store = load_vector_store()
    results = vector_store.similarity_search(question, k=k)
    return results