import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
DATASET_FILE = str(BASE_DIR / "datasets" / "eduagent_dataset.csv")
CLASSIFIER_PATH = str(BASE_DIR / "models" / "distilbert_eduagent_v2")
DB_FILE = os.getenv("EDUAGENT_DB_FILE", str(BASE_DIR / "runtime" / "eduagent_app.db"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

CLASSIFIER_LABELS = ["beginner", "intermediate", "advanced"]

# RAG retrieval sizes
RAG_TOP_N = int(os.getenv("RAG_TOP_N", "3"))           # question-driven examples
RAG_WEAK_TOP_N = int(os.getenv("RAG_WEAK_TOP_N", "2")) # weak-area targeted examples
