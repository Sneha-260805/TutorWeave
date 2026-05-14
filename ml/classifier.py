import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.settings import CLASSIFIER_PATH, CLASSIFIER_HF_REPO, CLASSIFIER_LABELS

logger = logging.getLogger(__name__)

tokenizer = None
model = None
_classifier_available = False
_classifier_source = "none"


def _try_load(path_or_repo: str, label: str):
    """Attempt to load tokenizer + model from a path or HF repo ID."""
    tok = AutoTokenizer.from_pretrained(path_or_repo)
    mdl = AutoModelForSequenceClassification.from_pretrained(path_or_repo)
    mdl.eval()
    return tok, mdl


# ── Step 1: Local path ────────────────────────────────────────────────────────
_local_path = Path(CLASSIFIER_PATH)
_has_local_weights = (
    (_local_path / "pytorch_model.bin").exists()
    or (_local_path / "model.safetensors").exists()
)

if _has_local_weights:
    try:
        tokenizer, model = _try_load(CLASSIFIER_PATH, "local")
        _classifier_available = True
        _classifier_source = f"local ({CLASSIFIER_PATH})"
        logger.info("CLASSIFIER SOURCE: local path  →  %s", CLASSIFIER_PATH)
    except Exception as _local_err:
        logger.warning("Local classifier load failed: %s", _local_err)
else:
    logger.info(
        "Local model weights not found at %s — will try HuggingFace Hub.",
        CLASSIFIER_PATH,
    )

# ── Step 2: HuggingFace Hub (if local unavailable) ────────────────────────────
if not _classifier_available:
    try:
        logger.info("CLASSIFIER SOURCE: loading from HuggingFace Hub  →  %s", CLASSIFIER_HF_REPO)
        tokenizer, model = _try_load(CLASSIFIER_HF_REPO, "HF Hub")
        _classifier_available = True
        _classifier_source = f"HuggingFace Hub ({CLASSIFIER_HF_REPO})"
        logger.info("Classifier successfully loaded from HuggingFace Hub: %s", CLASSIFIER_HF_REPO)
    except Exception as _hf_err:
        logger.warning("HuggingFace Hub classifier load failed: %s", _hf_err)
        logger.warning(
            "CLASSIFIER SOURCE: heuristics-only — "
            "both local path and HF Hub failed.  Level detection may be less accurate."
        )

if not _classifier_available:
    tokenizer = None
    model = None

# Confidence below this → trust argmax but also run heuristics.
# We no longer hard-force "intermediate" for ambiguous cases.
LOW_CONFIDENCE_THRESHOLD = 0.55
# Below this → truly uncertain, default to intermediate.
VERY_LOW_CONFIDENCE_THRESHOLD = 0.32

ADVANCED_INTENT_MARKERS = (
    "analyze", "analyse",
    "derive", "derivation",
    "prove", "proof",
    "compare and contrast",
    "in terms of",           # comparison questions: "compare X with Y in terms of Z"
    "optimize", "optimise",
    "convergence", "converge",
    "theorem",
    "architecture",
    "attention mechanism",
    "training objective",
    "scaling law",
    "fine tuning", "fine-tuning", "finetuning",
    "backpropagation", "backprop",
    "gradient flow",
    "regularization", "regularisation",
    "hyperparameter tuning", "hyperparameter optimization",
    "batch normalization", "layer normalization",
    "mathematically", "formally", "theoretically",
    "from scratch",
    "time complexity", "space complexity",
    "trade-off", "tradeoff",
    "loss landscape",
    "vanishing gradient", "exploding gradient",
    "saddle point",
    "weight initialization",
    "learning rate schedule",
    "curriculum learning",
    "multi-head attention",
    "positional encoding",
    "knowledge distillation",
    "quantization",
    "pruning",
)

BEGINNER_INTENT_PREFIXES = (
    "what is ",
    "what are ",
    "define ",
    "tell me about ",
    "what does ",
    "can you explain ",
    "i am new to",
    "i'm new to",
    "just started",
    "getting started with",
    "for beginners",
    "beginner guide",
    "simple explanation",
    "easy explanation",
    "eli5",  # explain like I'm 5
)

BEGINNER_INTENT_PHRASES = (
    "in simple terms",
    "simply",
    "basic explanation",
    "beginner",
    "for a beginner",
    "easy way to understand",
    "simple way",
    "layman",
)

COMPARISON_INTENT_MARKERS = (
    "compare ",
    "difference between",
    "differentiate between",
    "distinguish between",
    " vs ",
    " versus ",
)

ADVANCED_COMPARISON_MARKERS = (
    "derive",
    "derivation",
    "prove",
    "proof",
    "theorem",
    "convergence",
    "converge",
    "mathematically",
    "formally",
    "theoretically",
    "time complexity",
    "space complexity",
    "loss landscape",
    "bias correction",
    "generalization",
    "trade-off",
    "tradeoff",
)


def _has_advanced_intent(text_lower: str) -> bool:
    return any(marker in text_lower for marker in ADVANCED_INTENT_MARKERS)


def _has_comparison_intent(text_lower: str) -> bool:
    padded = f" {text_lower} "
    return any(marker in padded for marker in COMPARISON_INTENT_MARKERS)


def _has_advanced_comparison_intent(text_lower: str) -> bool:
    return any(marker in text_lower for marker in ADVANCED_COMPARISON_MARKERS)


def _has_beginner_intent(text_lower: str) -> bool:
    # Explicit beginner phrases always win
    if any(phrase in text_lower for phrase in BEGINNER_INTENT_PHRASES):
        return True

    tokens = text_lower.split()
    word_count = len(tokens)

    # "what is X" / "what are X" / "define X" are beginner only for short questions
    # (long ones like "what is the difference between X and Y" are intermediate)
    simple_prefixes = ("what is ", "what are ", "define ")
    if any(text_lower.startswith(p) for p in simple_prefixes) and word_count <= 5:
        return True

    # Longer explicit beginner openers
    explicit_beginner = (
        "tell me about ", "can you explain ",
        "i am new to", "i'm new to", "just started",
        "getting started with", "for beginners",
        "beginner guide", "simple explanation",
        "easy explanation", "eli5",
    )
    if any(text_lower.startswith(p) or p in text_lower for p in explicit_beginner):
        return True

    # Short "explain X" without advanced terms
    if text_lower.startswith("explain ") and word_count <= 6:
        return True

    return False


def _vocabulary_complexity(text_lower: str) -> str | None:
    """
    Return 'advanced' or 'beginner' if vocabulary strongly signals a level,
    or None if inconclusive.
    """
    words = text_lower.split()
    word_count = len(words)

    # Long questions with multiple technical terms → advanced
    tech_term_count = sum(1 for m in ADVANCED_INTENT_MARKERS if m in text_lower)
    if word_count >= 15 and tech_term_count >= 2:
        return "advanced"
    if word_count >= 20 and tech_term_count >= 1:
        return "advanced"

    # Very short simple "what is X" with ≤ 2 content words → beginner
    if word_count <= 5 and text_lower.startswith(("what is", "what are", "define")):
        return "beginner"

    return None


def _heuristic_predict(text_lower: str):
    word_count = len(text_lower.split())
    if _has_comparison_intent(text_lower) and not _has_advanced_comparison_intent(text_lower):
        return "intermediate", [0.1, 0.8, 0.1]
    if _has_advanced_intent(text_lower):
        return "advanced", [0.05, 0.15, 0.8]
    vocab_sig = _vocabulary_complexity(text_lower)
    if vocab_sig == "advanced":
        return "advanced", [0.05, 0.2, 0.75]
    if _has_beginner_intent(text_lower) and word_count <= 10:
        return "beginner", [0.8, 0.15, 0.05]
    if vocab_sig == "beginner":
        return "beginner", [0.75, 0.2, 0.05]
    return "intermediate", [0.1, 0.8, 0.1]


def predict_level(text):
    """
    Predict the learner difficulty level.

    Primary signal: DistilBERT classifier.
    Overrides:
    - If max confidence < VERY_LOW_CONFIDENCE_THRESHOLD → default intermediate.
    - If LOW ≤ confidence < threshold → trust argmax (no longer forced to intermediate).
    - Beginner intent heuristic (short, simple phrasing, no advanced terms).
    - Advanced intent heuristic (explicit advanced vocabulary).
    - Vocabulary complexity signal for edge cases.
    """
    text_lower = str(text).lower().strip()

    if not _classifier_available:
        return _heuristic_predict(text_lower)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs.tolist()[0]
    predicted_label = CLASSIFIER_LABELS[pred_idx]
    max_conf = max(confidence)

    if max_conf < VERY_LOW_CONFIDENCE_THRESHOLD:
        # Truly uncertain — fall back to intermediate but allow heuristics below
        predicted_label = "intermediate"
    # Between VERY_LOW and LOW_CONFIDENCE_THRESHOLD: keep argmax (no longer forcing intermediate)

    word_count = len(text_lower.split())

    # Heuristic overrides — applied after classifier
    if _has_comparison_intent(text_lower) and not _has_advanced_comparison_intent(text_lower):
        predicted_label = "intermediate"
    elif _has_advanced_intent(text_lower):
        predicted_label = "advanced"
    elif _has_beginner_intent(text_lower) and not _has_advanced_intent(text_lower) and word_count <= 8:
        predicted_label = "beginner"
    else:
        vocab_sig = _vocabulary_complexity(text_lower)
        if vocab_sig and max_conf < LOW_CONFIDENCE_THRESHOLD:
            predicted_label = vocab_sig

    return predicted_label, confidence
