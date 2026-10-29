import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_FILE = DATA_DIR / "documents.json"
CHUNKS_FILE = DATA_DIR / "chunks.json"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "140"))
TOP_K = int(os.getenv("TOP_K", "5"))
