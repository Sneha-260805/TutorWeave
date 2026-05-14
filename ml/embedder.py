"""
Shared sentence-transformers embedding model.

Loaded once at import time and reused by both ml/retriever.py and
ml/topic_detector.py — avoids loading a 90 MB model twice.
"""
import logging
import os

# Must be set before transformers is imported.
os.environ.setdefault("USE_TF", "0")

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
    semantic_available = True
    logger.info("Embedder loaded from local cache: all-MiniLM-L6-v2")
except Exception as _err:
    embed_model = None
    semantic_available = False
    logger.warning("sentence-transformers not available (%s). Semantic features disabled.", _err)
