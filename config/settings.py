import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
DATASET_FILE = str(BASE_DIR / "datasets" / "data_easy" / "eduagent_dataset_easy_v2.csv")
CLASSIFIER_PATH = str(BASE_DIR / "models" / "distilbert_eduagent_v2")
CLASSIFIER_HF_REPO = os.getenv("CLASSIFIER_HF_REPO", "SSneha2005/Eduagent_distilbert")
DB_FILE = os.getenv("EDUAGENT_DB_FILE", str(BASE_DIR / "runtime" / "eduagent_app.db"))
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "8"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

CLASSIFIER_LABELS = ["beginner", "intermediate", "advanced"]

# RAG retrieval sizes
RAG_TOP_N = int(os.getenv("RAG_TOP_N", "3"))           #     question-driven examples
RAG_WEAK_TOP_N = int(os.getenv("RAG_WEAK_TOP_N", "2")) # weak-area targeted examples

# Evaluation settings
EVAL_MODEL = os.getenv("EVAL_MODEL", "gemini-2.0-flash")
N_EVAL_SAMPLES = int(os.getenv("N_EVAL_SAMPLES", "5"))
