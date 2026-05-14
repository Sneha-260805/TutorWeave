"""
Semantic retrieval using sentence-transformers (all-MiniLM-L6-v2).

Each unique (level, topic) slice of the dataset is encoded once and cached in
memory. At query time only the user question is encoded — O(1) model calls per
query regardless of dataset size.

Falls back to TF-IDF cosine similarity if sentence-transformers is not
installed, so the app still works without the extra dependency.
"""
import logging
import os

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from config.settings import DATASET_FILE
from ml.topic_detector import clean_text, detect_best_topic, expand_topic_aliases
from ml.embedder import embed_model as _embed_model, semantic_available as _semantic_available

logger = logging.getLogger(__name__)

if _semantic_available:
    logger.info("Semantic retrieval: using shared all-MiniLM-L6-v2 embedder.")
else:
    logger.warning("sentence-transformers not available. Retrieval falling back to TF-IDF.")

_KEYWORD_STOPS = frozenset({
    "what", "is", "are", "how", "the", "a", "an", "in", "of", "to",
    "and", "for", "with", "does", "do", "can", "i", "me", "my", "it",
    "this", "that", "be", "at", "by", "on", "or", "as", "its", "was",
    "why", "when", "which", "where", "who",
})


def _ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _keyword_overlap(query: str, candidates: list[str]) -> np.ndarray:
    """
    Unigram + bigram Jaccard similarity on content words (stop-words removed).
    Including bigrams lets the exact source question rank above near-duplicates
    that share topic vocabulary but not the same phrase structure.
    """
    q_tokens = [w for w in clean_text(query).split() if w not in _KEYWORD_STOPS]
    if not q_tokens:
        return np.zeros(len(candidates))

    q_ngrams = set(q_tokens) | _ngrams(q_tokens, 2)

    scores = []
    for c in candidates:
        c_tokens = [w for w in clean_text(c).split() if w not in _KEYWORD_STOPS]
        c_ngrams = set(c_tokens) | _ngrams(c_tokens, 2)
        union = q_ngrams | c_ngrams
        scores.append(len(q_ngrams & c_ngrams) / len(union) if union else 0.0)
    return np.array(scores, dtype=float)

# ---------------------------------------------------------------------------
# Dataset — loaded once at import, shared with tutor_agent via `df`
# ---------------------------------------------------------------------------
df = pd.read_csv(DATASET_FILE)
# ---------------------------------------------------------------------------
# Index cache  {(n_rows, level, topic): (filtered_df, vectors)}
# ---------------------------------------------------------------------------
_INDEX_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Helpers shared with both backends
# ---------------------------------------------------------------------------

def filter_by_level(df_in: pd.DataFrame, level: str) -> pd.DataFrame:
    # Thresholds calibrated to v2 dataset percentiles (p5–p97 per level).
    # Beginner: min=57, p95=100, max=114  → keep < 120
    # Intermediate: min=78, p95=144, max=170 → keep 40–165
    # Advanced: min=102, all answers long → keep > 80 (unchanged)
    df_in = df_in.copy()
    df_in["answer_length"] = df_in["answer"].apply(lambda x: len(str(x).split()))
    if level == "beginner":
        return df_in[df_in["answer_length"] < 120]
    if level == "intermediate":
        return df_in[(df_in["answer_length"] >= 40) & (df_in["answer_length"] <= 165)]
    if level == "advanced":
        return df_in[df_in["answer_length"] > 80]
    return df_in


def _complexity_penalty(text: str) -> float:
    text = clean_text(text)
    penalty = 0.0
    hard_terms = {
        "derive", "proof", "prove", "theorem", "convergence",
        "subgradient", "vanishing gradient", "quasi newton",
        "non differentiable", "high dimensional",
    }
    if len(text.split()) > 18:
        penalty += 0.15
    for term in hard_terms:
        if term in text:
            penalty += 0.15
    return penalty


def _cache_key(level: str, topic: str | None) -> tuple:
    return (len(df), level.strip().lower(), (topic or "").strip().lower())


def _filter_slice(level: str, topic: str | None) -> pd.DataFrame:
    filtered = df[df["level"].astype(str).str.lower() == level].copy()
    if topic:
        # Try subtopic match first (more precise), then fall back to topic
        if "subtopic" in df.columns:
            by_subtopic = filtered[
                filtered["subtopic"].astype(str).str.lower().str.contains(
                    topic.lower(), na=False
                )
            ]
            if len(by_subtopic) >= 3:   # only use subtopic if enough results
                filtered = by_subtopic
            else:
                by_topic = filtered[
                    filtered["topic"].astype(str).str.lower() == topic.lower()
                ]
                if len(by_topic) > 0:
                    filtered = by_topic
        else:
            by_topic = filtered[
                filtered["topic"].astype(str).str.lower() == topic.lower()
            ]
            if len(by_topic) > 0:
                filtered = by_topic

    filtered = filter_by_level(filtered, level)
    if len(filtered) == 0:
        # Relax length filter and retry
        filtered = df[df["level"].astype(str).str.lower() == level].copy()
        if topic:
            by_topic = filtered[
                filtered["topic"].astype(str).str.lower() == topic.lower()
            ]
            if len(by_topic) > 0:
                filtered = by_topic
    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Semantic backend (sentence-transformers)
# ---------------------------------------------------------------------------

def _build_semantic_index(level: str, topic: str | None):
    filtered = _filter_slice(level, topic)
    if len(filtered) == 0:
        return filtered, None
    questions = filtered["question"].tolist()
    vectors = _embed_model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    filtered["penalty"] = filtered["question"].apply(_complexity_penalty)
    return filtered, vectors


def _semantic_retrieve(user_question: str, level: str, topic: str | None, top_n: int):
    key = _cache_key(level, topic)
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _build_semantic_index(level, topic)
    filtered, vectors = _INDEX_CACHE[key]

    if len(filtered) == 0 or vectors is None:
        return pd.DataFrame(columns=["question", "answer", "level", "topic"])

    user_vec = _embed_model.encode([user_question], convert_to_numpy=True)
    sims = cosine_similarity(user_vec, vectors).flatten()

    # Hybrid score: 80% semantic cosine + 20% keyword Jaccard – penalty
    kw_scores = _keyword_overlap(user_question, filtered["question"].tolist())
    scored = filtered.copy()
    scored["similarity"] = sims
    scored["keyword"] = kw_scores
    scored["final_score"] = 0.75 * scored["similarity"] + 0.25 * scored["keyword"] - scored["penalty"]
    scored = scored.sort_values("final_score", ascending=False)
    return scored.head(top_n)[["question", "answer", "level", "topic"]]


# ---------------------------------------------------------------------------
# TF-IDF fallback backend
# ---------------------------------------------------------------------------

def _build_tfidf_index(level: str, topic: str | None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    filtered = _filter_slice(level, topic)
    if len(filtered) == 0:
        return filtered, None, None
    filtered["clean_question"] = filtered["question"].apply(clean_text)
    filtered["penalty"] = filtered["question"].apply(_complexity_penalty)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(filtered["clean_question"].tolist())
    return filtered, vectorizer, matrix


def _tfidf_retrieve(user_question: str, level: str, topic: str | None, top_n: int):
    key = ("tfidf", *_cache_key(level, topic))
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = _build_tfidf_index(level, topic)
    filtered, vectorizer, matrix = _INDEX_CACHE[key]

    if len(filtered) == 0 or vectorizer is None:
        return pd.DataFrame(columns=["question", "answer", "level", "topic"])

    user_vec = vectorizer.transform([expand_topic_aliases(clean_text(user_question))])
    sims = cosine_similarity(user_vec, matrix).flatten()

    scored = filtered.copy()
    scored["similarity"] = sims
    scored["final_score"] = scored["similarity"] - scored["penalty"]
    scored = scored.sort_values("final_score", ascending=False)
    return scored.head(top_n)[["question", "answer", "level", "topic"]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_for_weak_areas(
    weak_concepts: list,
    topic: str,
    level: str,
    top_n: int = 2,
    exclude_questions: set = None,
) -> pd.DataFrame:
    """
    Retrieve dataset examples most relevant to the learner's recorded weak concepts.

    Each weak concept string is used as a query against the same semantic index
    used for normal retrieval. Results across all concepts are pooled, de-duplicated,
    re-ranked by best similarity, and the top-N returned.

    exclude_questions: set of question strings already returned by retrieve_examples,
    so we don't repeat the same rows.
    """
    if not weak_concepts:
        return pd.DataFrame(columns=["question", "answer", "level", "topic"])

    exclude_questions = exclude_questions or set()
    level = str(level).strip().lower()

    all_scored = []

    for concept in weak_concepts:
        query = f"{concept} {topic}".strip()
        try:
            if _semantic_available:
                result = _semantic_retrieve(query, level, topic, top_n=top_n + 2)
            else:
                result = _tfidf_retrieve(query, level, topic, top_n=top_n + 2)
            if len(result) > 0:
                all_scored.append(result)
        except Exception:
            continue

    if not all_scored:
        return pd.DataFrame(columns=["question", "answer", "level", "topic"])

    pooled = pd.concat(all_scored, ignore_index=True)
    pooled = pooled.drop_duplicates(subset=["question"])
    pooled = pooled[~pooled["question"].isin(exclude_questions)]

    if len(pooled) == 0:
        return pd.DataFrame(columns=["question", "answer", "level", "topic"])

    return pooled.head(top_n)[["question", "answer", "level", "topic"]]


def retrieve_examples(user_question: str, level: str, top_n: int = 2) -> pd.DataFrame:
    """
    Retrieve the top-N most semantically similar examples for the given question.

    Uses sentence-transformers (all-MiniLM-L6-v2) dense embeddings when
    available, falling back to TF-IDF cosine similarity otherwise.
    """
    level = str(level).strip().lower()
    topic = detect_best_topic(user_question, level, df)

    if _semantic_available:
        results = _semantic_retrieve(user_question, level, topic, top_n)
    else:
        results = _tfidf_retrieve(user_question, level, topic, top_n)

    if len(results) == 0:
        fallback = df[df["level"].astype(str).str.lower() == level]
        results = fallback.sample(min(top_n, len(fallback)), random_state=42)[
            ["question", "answer", "level", "topic"]
        ]

    return results
