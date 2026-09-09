from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil
import os

from src.config import PDF_FOLDER, DB_PATH, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
def load_pdfs(pdf_folder):
    "Fonction qui charge tous les PDF du dossier et retourne la liste des documents ( métadonnées : fichier + page )."
    documents = []
    pdf_files = list(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {pdf_folder}")

    for pdf_file in pdf_files:
        print(f"Chargement : {pdf_file.name}")
        loader = PDFPlumberLoader(str(pdf_file))
        documents.extend(loader.load())
    return documents

def split_documents(documents):
    "Fonction qui découpe les documents en petits morceaux (chunks) pour faciliter la recherche."
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

    )
    return splitter.split_documents(documents)

def build_vector_store(chunks):
    "Fonction qui calcule les embeddings et les stocke dans chromaDB."
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH,
    )

    return vector_store

def main():
    documents = load_pdfs(PDF_FOLDER)
    print(f"{len(documents)}pages chargées.")

    chunks = split_documents(documents)
    print(f"{len(chunks)} chunks crées.")

    build_vector_store(chunks)
    print(f"Index vectoriel créé et sauvegardé dans : {DB_PATH}")

def rebuild_index():
    """Vide complètement l'index existant et le reconstruit à partir des PDF restants dans le dossier."""
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))
    if not pdf_files:
        return  # plus aucun PDF, on laisse l'index vide

    documents = load_pdfs(PDF_FOLDER)
    chunks = split_documents(documents)
    build_vector_store(chunks)

if __name__ == "__main__":
    main()