import logging
import os

# Disable TF integration before transformers is imported to avoid
# Keras 3 / tf-keras compatibility errors on systems without TF installed.
os.environ.setdefault("USE_TF", "0")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config.settings import CLASSIFIER_PATH, CLASSIFIER_LABELS

logger = logging.getLogger(__name__)

try:
    # AutoTokenizer reads tokenizer_config.json and picks the right class
    # automatically — avoids the BertTokenizer vs DistilBertTokenizer mismatch
    # that occurs when the checkpoint was saved with BertTokenizer.
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH)
    model.eval()
    _classifier_available = True
    logger.info("Classifier loaded from %s.", CLASSIFIER_PATH)
except Exception as _load_err:
    logger.warning("Could not load classifier from %s: %s. Falling back to heuristics only.", CLASSIFIER_PATH, _load_err)
    tokenizer = None
    model = None
    _classifier_available = False

LOW_CONFIDENCE_THRESHOLD = 0.6
ADVANCED_INTENT_MARKERS = (
    "analyze",
    "derive",
    "prove",
    "compare",
    "optimize",
    "convergence",
    "theorem",
    "architecture",
    "attention mechanism",
    "training objective",
    "scaling law",
    "fine tuning",
    "backpropagation",
)
BEGINNER_INTENT_PREFIXES = (
    "what is ",
    "what are ",
    "define ",
    "tell me about ",
)
BEGINNER_INTENT_PHRASES = (
    "in simple terms",
    "simply",
    "basic explanation",
    "beginner",
)


def _has_advanced_intent(text_lower: str) -> bool:
    return any(marker in text_lower for marker in ADVANCED_INTENT_MARKERS)


def _has_beginner_intent(text_lower: str) -> bool:
    if any(text_lower.startswith(prefix) for prefix in BEGINNER_INTENT_PREFIXES):
        return True
    if any(phrase in text_lower for phrase in BEGINNER_INTENT_PHRASES):
        return True
    tokens = text_lower.split()
    return text_lower.startswith("explain ") and len(tokens) <= 6


def _heuristic_predict(text_lower: str):
    word_count = len(text_lower.split())
    if _has_advanced_intent(text_lower):
        return "advanced", [0.05, 0.15, 0.8]
    if _has_beginner_intent(text_lower) and word_count <= 8:
        return "beginner", [0.8, 0.15, 0.05]
    return "intermediate", [0.1, 0.8, 0.1]


def predict_level(text):
    """
    Predict the learner difficulty level.

    The trained classifier is the primary signal. The beginner-intent heuristic
    only overrides the classifier for short questions (<=8 words) that don't
    contain advanced domain markers — this prevents "what is the convergence rate
    of Adam?" from being mis-classified as beginner.
    """
    text_lower = str(text).lower().strip()

    if not _classifier_available:
        return _heuristic_predict(text_lower)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    # DistilBERT has no segment embeddings — drop token_type_ids if the
    # tokenizer (BertTokenizer) added them.
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = F.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs.tolist()[0]
    predicted_label = CLASSIFIER_LABELS[pred_idx]

    if max(confidence) < LOW_CONFIDENCE_THRESHOLD:
        predicted_label = "intermediate"

    word_count = len(text_lower.split())
    if _has_beginner_intent(text_lower) and not _has_advanced_intent(text_lower) and word_count <= 8:
        predicted_label = "beginner"

    return predicted_label, confidence
