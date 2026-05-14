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
        "feature engineering",
        "feature selection",
        "cross validation",
        "train test split",
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
        "activation function",
        "relu",
        "sigmoid",
        "neuron",
        "hidden layer",
        "neural network weights",
        "network bias",
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
        "pre training",
        "pretraining",
        "language modeling",
        "token prediction",
        "autoregressive",
        "causal language model",
    ],
    "natural language processing": [
        "nlp",
        "natural language processing",
        "text processing",
        "language model",
        "tokenization",
        "tokenizer",
        "tokenize",
        "sentiment analysis",
        "sentiment classification",
        "named entity recognition",
        "ner",
        "text classification",
        "text mining",
        "part of speech",
        "pos tagging",
        "word embedding",
        "word2vec",
        "glove",
        "text generation",
        "machine translation",
        "question answering",
        "information extraction",
        "coreference",
        "dependency parsing",
        "stemming",
        "lemmatization",
        "stop words",
        "bag of words",
        "tf idf",
        "tfidf",
    ],
    "retrieval augmented generation": [
        "rag",
        "retrieval augmented generation",
        "retrieval-augmented generation",
        "retrieval augmentation",
        "document retrieval",
        "knowledge retrieval",
        "grounded generation",
    ],
    "vector databases": [
        "vector db",
        "vector database",
        "vector databases",
        "embedding database",
        "embedding store",
        "faiss",
        "chromadb",
        "pinecone",
        "weaviate",
        "milvus",
        "qdrant",
        "ann",
        "approximate nearest neighbor",
        "approximate nearest neighbour",
        "vector search",
        "similarity search",
        "vector index",
        "vector indexing",
        "hnsw",
        "ivf",
        "vector store",
    ],
    "convolutional neural networks": [
        "cnn",
        "cnns",
        "convolutional neural network",
        "convolutional neural networks",
        "convolution",
        "convolutional",
        "convolutional layer",
        "feature map",
        "feature maps",
        "pooling layer",
        "max pooling",
        "average pooling",
        "stride",
        "padding",
        "kernel size",
        "filter",
        "depthwise convolution",
        "1x1 convolution",
    ],
    "recurrent neural networks": [
        "rnn",
        "rnns",
        "recurrent neural network",
        "recurrent neural networks",
        "lstm",
        "gru",
        "long short term memory",
        "gated recurrent",
        "vanishing gradient rnn",
        "sequence model",
        "time series model",
        "formal grammar",
        "formal grammars",
        "regular language",
        "regular languages",
        "context free grammar",
        "context-free",
        "chomsky hierarchy",
    ],
    "generative adversarial networks": [
        "gan",
        "gans",
        "generative adversarial network",
        "generative adversarial networks",
        "discriminator",
        "generator network",
        "mode collapse",
        "wasserstein",
        "image synthesis",
        "image generation",
    ],
    "transformer models": [
        "transformer model",          # "transformers" (plural, bare) removed — too ambiguous
        "transformer architecture",
        "bert",
        "gpt model",
        "self attention",
        "encoder decoder",
        "positional encoding",
        "attention head",
        "feed forward network transformer",
        "layer normalization",
        "t5",
        "roberta",
        "deberta",
        "transformer based",
        "transformer network",
    ],
    "gradient descent": [
        "learning rate",
        "optimizer",
        "optimisation",
        "optimization",
        "sgd",
        "stochastic gradient descent",
        "mini batch",
        "adam optimizer",
        "momentum",
        "weight update",
        "loss minimization",
        "convergence",
        "gradient clipping",
        "learning rate schedule",
        "warmup",
        "cosine annealing",
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
        "generalization",
        "early stopping",
        "data augmentation",
        "noise injection",
    ],
    "transfer learning": [
        "transfer learning",
        "fine tuning",
        "fine-tuning",
        "finetuning",
        "finetune",
        "pretrained",
        "pre-trained",
        "pretrain",
        "domain adaptation",
        "downstream task",
        "source domain",
        "target domain",
        "feature extraction",
        "frozen layers",
        "freeze layers",
        "lora",
        "peft",
        "adapter",
        "prompt tuning",
        "instruction tuning",
        "zero shot transfer",
        "few shot transfer",
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
        "policy",
        "value function",
        "exploration exploitation",
        "ppo",
        "dqn",
        "actor critic",
        "reward shaping",
        "rlhf",
        "reinforcement learning from human feedback",
    ],
    "agentic ai": [
        "agentic", "ai agent", "ai agents", "autonomous agent",
        "autonomous ai", "tool use", "function calling",
        "multi agent", "agent loop", "react agent", "chain of agents",
        "tool calling", "agentic workflow", "llm agent",
        "memory agent", "task planning", "agent reasoning",
        "agent framework", "plan and execute", "agent observation",
        "agent action", "agent tool", "agentic system",
    ],
    "attention mechanism": [
        "attention", "self-attention", "multi-head attention",
        "scaled dot product", "query key value", "qkv",
        "cross attention", "attention weight", "attention score",
        "softmax attention", "key query value", "attention map",
        "context vector", "attention computation",
        "dot product attention", "additive attention",
        "attention head", "causal attention", "masked attention",
    ],
    "backpropagation": [
        "backpropagation", "backprop", "back propagation",
        "chain rule", "gradient flow",
        "backward pass", "gradient computation", "computational graph",
        "automatic differentiation", "autograd",
        "gradient backflow", "gradient signal",
        "neural network training gradient", "loss gradient",
        "weight gradient",
    ],
    "classification": [
        "classify", "classifier", "logistic regression",
        "binary classification", "multiclass", "multi class",
        "softmax classifier", "support vector machine", "svm",
        "naive bayes", "knn", "k nearest neighbor",
        "class prediction", "discriminative model",
        "label assignment", "multi-label classification",
        "imbalanced classes", "class weight",
        "one vs rest", "one vs all", "class probabilities",
    ],
    "clustering algorithms": [
        "clustering", "k-means", "kmeans", "dbscan",
        "hierarchical clustering", "unsupervised clustering",
        "gaussian mixture", "em algorithm", "silhouette",
        "k means clustering", "elbow method", "centroid",
        "cluster assignment", "density estimation",
        "agglomerative clustering", "divisive clustering",
        "cluster analysis", "cluster quality",
    ],
    "computer vision": [
        "cv", "image recognition", "object detection",
        "image classification", "vision model",
        "semantic segmentation", "instance segmentation",
        "yolo", "resnet", "vgg", "mobilenet",
        "bounding box", "anchor box",
        "image preprocessing", "feature descriptor",
        "visual representation", "patch embedding",
        "vision transformer", "vit", "image encoder",
        "keypoint detection", "depth estimation",
    ],
    "decision trees": [
        "decision tree", "random forest", "xgboost",
        "gradient boosting", "gini impurity", "information gain",
        "cart", "entropy split", "ensemble method",
        "bagging", "boosting",
        "tree pruning", "decision node", "leaf node",
        "splitting criterion", "regression tree",
        "tree depth", "feature importance", "lightgbm",
    ],
    "graph neural networks": [
        "gnn", "gnns", "graph neural network",
        "graph convolution", "message passing",
        "node classification", "link prediction", "graph embedding",
        "node embedding", "edge features", "graph attention",
        "neighbor aggregation", "spectral graph",
        "gcn", "graphsage", "graph sage",
        "heterogeneous graph", "knowledge graph",
    ],
    "model evaluation metrics": [
        "accuracy", "precision", "recall", "f1 score", "f1",
        "roc", "auc", "confusion matrix", "evaluation metric",
        "mean average precision", "map score", "bleu", "rouge",
        "perplexity", "validation loss", "test accuracy",
        "performance metric", "model performance",
        "log loss", "cross entropy loss",
        "mean squared error", "mse",
        "mean absolute error", "mae", "r squared",
        "model benchmarking", "error analysis",
    ],
    "prompt engineering": [
        "prompt", "prompting", "few shot", "zero shot",
        "chain of thought", "cot", "system prompt",
        "in context learning", "instruction following",
        "prompt template", "prompt design",
        "prompt chaining", "prompt injection",
        "role prompting", "structured prompting",
        "prompt optimization", "output formatting",
        "meta prompt", "prompt tuning",
    ],
    "ethical ai and bias": [
        "ethical ai", "ai ethics", "responsible ai",
        "fairness", "ai bias", "algorithmic bias",
        "bias in ai", "bias in ml",
        "explainability", "xai", "interpretable",
        "interpretability", "transparency",
        "accountability", "ai safety",
        "bias mitigation", "demographic parity",
        "equalized odds", "disparate impact",
        "model fairness", "model transparency",
        "black box", "model interpretability",
        "ethical concerns", "ethical issues",
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
        best_score = float(sims[best_idx])
        logger.debug(
            "Semantic topic detection: '%s' → '%s' (score=%.3f)",
            user_question, topic_names[best_idx], best_score,
        )
        # Only accept a semantic match when the score is convincing enough;
        # a very low score means no topic is a good match → use the alias
        # default rather than a noisy winner.
        if best_score < 0.15:
            return _default_topic(df)
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
