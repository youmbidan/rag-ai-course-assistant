from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_FOLDER = BASE_DIR /"data" / "pdfs"
DB_PATH = str(BASE_DIR / "chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4