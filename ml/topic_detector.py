import re
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.embedder import embed_model as _embed_model, semantic_available as _semantic_available

logger = logging.getLogger(__name__)

_TOPIC_INDEX_CACHE = {}
TOPIC_ALIASES = {
    "machine learning basics": [
        "ml",
        "machine learning",
        "types of ml",
        "types of machine learning",
        "ml types",
        "categories of ml",
        "machine learning categories",
        "machine learning algorithms",
        "ml algorithms",
        "supervised",
        "unsupervised",
        "semi supervised",
        "what is ml",
        "what is machine learning",
    ],
    "neural networks": [
        "deep learning",
        "dl",
        "artificial neural network",
        "ann",
        "neural net",
        "neural nets",
        "perceptron",
        "multilayer perceptron",
        "mlp",
    ],
    "large language models": [
        "llm",
        "llms",
        "large language model",
        "large language models",
        "gpt",
        "chatgpt",
        "foundation model",
        "foundation models",
    ],
    "natural language processing": [
        "nlp",
        "natural language processing",
        "text processing",
        "language model",
    ],
    "retrieval augmented generation": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
    ],
    "vector databases": [
        "vector db",
        "vector database",
        "vector databases",
        "embedding database",
        "faiss",
        "chromadb",
        "pinecone",
    ],
    "convolutional neural networks": [
        "cnn",
        "cnns",
        "convolutional neural network",
        "convolutional neural networks",
        "convolution",
    ],
    "recurrent neural networks": [
        "rnn",
        "rnns",
        "recurrent neural network",
        "recurrent neural networks",
        "lstm",
        "gru",
        "long short term memory",
    ],
    "generative adversarial networks": [
        "gan",
        "gans",
        "generative adversarial network",
        "generative adversarial networks",
    ],
    "transformer models": [
        "transformer",
        "transformers",
        "bert",
        "gpt model",
        "self attention",
        "encoder decoder",
    ],
    "gradient descent": [
        "learning rate",
        "optimizer",
        "optimisation",
        "optimization",
        "sgd",
        "stochastic gradient descent",
        "mini batch",
    ],
    "overfitting and regularization": [
        "overfitting",
        "underfitting",
        "regularization",
        "regularisation",
        "dropout",
        "l1",
        "l2",
        "weight decay",
        "bias variance",
        "bias-variance",
    ],
    "transfer learning": [
        "transfer learning",
        "fine tuning",
        "fine-tuning",
        "pretrained",
        "pre-trained",
        "pretrain",
    ],
    "reinforcement learning": [
        "rl",
        "reinforcement learning",
        "reward",
        "agent environment",
        "q learning",
        "policy gradient",
        "markov",
        "mdp",
    ],
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _available_topic_lookup(df: pd.DataFrame):
    return {str(topic).lower(): topic for topic in df["topic"].dropna().unique()}


def _alias_topic(user_question_clean: str, df: pd.DataFrame):
    topic_lookup = _available_topic_lookup(df)
    padded_question = f" {user_question_clean} "
    for canonical_topic, aliases in TOPIC_ALIASES.items():
        if canonical_topic not in topic_lookup:
            continue
        for alias in aliases:
            alias_clean = clean_text(alias)
            if f" {alias_clean} " in padded_question:
                return topic_lookup[canonical_topic]
    return None


def expand_topic_aliases(text: str) -> str:
    expanded_terms = []
    padded_text = f" {text} "
    for canonical_topic, aliases in TOPIC_ALIASES.items():
        canonical_clean = clean_text(canonical_topic)
        for alias in aliases:
            alias_clean = clean_text(alias)
            if re.search(rf"\b{re.escape(alias_clean)}\b", padded_text):
                expanded_terms.append(canonical_clean)
                break
    if not expanded_terms:
        return text
    return re.sub(r"\s+", " ", f"{text} {' '.join(sorted(set(expanded_terms)))}").strip()


def _df_cache_key(df: pd.DataFrame, level: str):
    return (
        id(df),
        len(df),
        str(level).strip().lower(),
        tuple(df.columns),
    )


# ---------------------------------------------------------------------------
# Semantic topic index  (sentence-transformers)
# ---------------------------------------------------------------------------

def _build_semantic_topic_index(level: str, df: pd.DataFrame):
    """
    For each topic encode a rich document:
      "<topic name>: <all questions for that topic>"
    Mean-pool into one vector per topic. Cached per (dataset, level).
    """
    level_df = df[df["level"].astype(str).str.lower() == level].copy()
    if len(level_df) == 0:
        return None

    topic_docs = (
        level_df.groupby("topic")["question"]
        .apply(lambda qs: " ".join(qs.tolist()))
        .reset_index()
    )
    # Prepend topic name so the embedding reflects both name and typical questions
    topic_docs["document"] = topic_docs.apply(
        lambda row: f"{row['topic']}: {row['question']}", axis=1
    )

    documents = topic_docs["document"].tolist()
    topic_names = topic_docs["topic"].tolist()
    vectors = _embed_model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
    return topic_names, vectors


# ---------------------------------------------------------------------------
# TF-IDF fallback topic index
# ---------------------------------------------------------------------------

def _build_tfidf_topic_index(level: str, df: pd.DataFrame):
    level_df = df[df["level"].astype(str).str.lower() == level].copy()
    if len(level_df) == 0:
        return None

    topic_texts = (
        level_df.groupby("topic")["question"]
        .apply(lambda qs: " ".join([clean_text(q) for q in qs]))
        .reset_index()
    )
    topic_texts["question"] = topic_texts.apply(
        lambda row: f"{clean_text(row['topic'])} {row['question']}", axis=1
    )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    topic_matrix = vectorizer.fit_transform(topic_texts["question"].tolist())
    return topic_texts, vectorizer, topic_matrix


def _get_topic_index(level: str, df: pd.DataFrame):
    level = str(level).strip().lower()
    key = _df_cache_key(df, level)
    if key not in _TOPIC_INDEX_CACHE:
        if _semantic_available:
            _TOPIC_INDEX_CACHE[key] = ("semantic", _build_semantic_topic_index(level, df))
        else:
            _TOPIC_INDEX_CACHE[key] = ("tfidf", _build_tfidf_topic_index(level, df))
    return _TOPIC_INDEX_CACHE[key]


# ---------------------------------------------------------------------------
# Public detection function
# ---------------------------------------------------------------------------

def detect_best_topic(user_question: str, level: str, df: pd.DataFrame) -> str:
    """
    Detect the closest topic for the user question.

    Priority:
    1. Alias table exact match (fast, handles abbreviations like ml, cnn, rag)
    2. Semantic dense-embedding cosine similarity (sentence-transformers)
       Falls back to TF-IDF if sentence-transformers is unavailable.
    3. Generic ML fallback when all scores are zero.
    """
    user_question_clean = clean_text(user_question)
    alias_topic = _alias_topic(user_question_clean, df)

    # Alias match is authoritative — no need to run the model
    if alias_topic is not None:
        return alias_topic

    backend, index = _get_topic_index(level, df)
    if index is None:
        return _default_topic(df)

    if backend == "semantic":
        topic_names, topic_vectors = index
        user_vec = _embed_model.encode([user_question], convert_to_numpy=True)
        sims = cosine_similarity(user_vec, topic_vectors).flatten()
        best_idx = int(np.argmax(sims))
        logger.debug(
            "Semantic topic detection: '%s' → '%s' (score=%.3f)",
            user_question, topic_names[best_idx], sims[best_idx],
        )
        return topic_names[best_idx]

    # TF-IDF fallback path
    topic_texts, vectorizer, topic_matrix = index
    user_question_expanded = expand_topic_aliases(user_question_clean)
    user_vec = vectorizer.transform([user_question_expanded])
    sims = cosine_similarity(user_vec, topic_matrix).flatten()
    scored = topic_texts.copy()
    scored["similarity"] = sims
    if scored["similarity"].max() <= 0:
        return _default_topic(df)
    best_row = scored.sort_values("similarity", ascending=False).iloc[0]
    return best_row["topic"]


def _default_topic(df: pd.DataFrame) -> str:
    """Return the most general topic when nothing else matches."""
    lookup = _available_topic_lookup(df)
    for candidate in ("machine learning basics", "neural networks", "classification"):
        if candidate in lookup:
            return lookup[candidate]
    topics = sorted(lookup.values())
    return topics[0] if topics else "general"


# Keep for backward compatibility (retriever imports this)
def get_topic_index(level, df):
    return _get_topic_index(level, df)
