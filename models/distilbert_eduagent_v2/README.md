# DistilBERT EduAgent v2 — Fine-Tuned Difficulty Classifier

This directory contains the fine-tuned DistilBERT model for predicting learner difficulty levels (beginner, intermediate, advanced) in AI/ML educational contexts.

## Model Overview

- **Base model**: `distilbert-base-uncased` (6 layers, 66M parameters)
- **Training dataset**: 2,400 balanced synthetic AI/ML Q&A pairs (800 per difficulty level)
- **Train/val/test split**: 1,920 / 240 / 240 samples
- **Test accuracy**: **97.92%** (+35.8 pp over keyword heuristic baseline of 62.1%)
- **Inference**: CPU-only, < 100 ms per prediction
- **Fine-tuning framework**: HuggingFace Transformers

## Files

```text
config.json                    ← Model config (vocab size, hidden dims, etc.)
model.safetensors             ← Model weights (safetensors format)
tokenizer_config.json         ← Tokenizer config
special_tokens_map.json       ← Special token mappings
vocab.txt                     ← Vocabulary
```

## Usage

### In EduAgent

The model is loaded automatically by `ml/classifier.py`:

```python
from ml.classifier import classify_difficulty

level, confidence = classify_difficulty("What is supervised learning?")
# Returns: ("beginner", [0.98, 0.01, 0.01])
```

### Standalone (HuggingFace Hub)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "Sneha-260805/distilbert-eduagent-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

inputs = tokenizer("What is a neural network?", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
predicted_level = logits.argmax(dim=-1).item()
# 0 = beginner, 1 = intermediate, 2 = advanced
```

## Performance

### Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | 97.92% |
| Precision (macro) | 0.979 |
| Recall (macro) | 0.979 |
| F1-score (macro) | 0.979 |

### Confusion Matrix (Test Set)

```
        Predicted Beginner  Intermediate  Advanced
Actual  
Beginner      79              1             0
Inter.         0              78             2
Advanced       0              3             77
```

### Comparison with Baseline

| Method | Accuracy | Notes |
|--------|----------|-------|
| Keyword heuristic | 62.1% | Rule-based fallback (hand-crafted difficulty keywords) |
| DistilBERT v2 | **97.92%** | Fine-tuned model with full context understanding |

## Training Details

- **Framework**: HuggingFace `Trainer`
- **Loss**: CrossEntropyLoss
- **Optimizer**: AdamW with weight decay
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Epochs**: 3
- **Warmup steps**: 100
- **Device**: CUDA (GPU) during training; CPU/GPU at inference

## Fallback Behavior

If this model directory is absent, `ml/classifier.py` automatically falls back to a lightweight keyword heuristic classifier:

```python
def fallback_classifier(text):
    """Keyword-based difficulty detection (62.1% accuracy)"""
    text_lower = text.lower()
    beginner_keywords = ["what is", "explain", "basics", "introduction", "simple"]
    intermediate_keywords = ["how to", "implement", "optimize", "design pattern"]
    advanced_keywords = ["edge case", "architectural", "performance tuning", "novel"]
    # ... scoring logic ...
```

This ensures the app remains functional even if model weights are unavailable, though with significantly lower accuracy.

## Integration with EduAgent

The classifier is called at the start of every tutor interaction:

1. User asks a question
2. `classify_difficulty(question)` returns `(level, [logits])`
3. Confidence scores drive memory hints (low confidence → more conservative teaching)
4. Level is recorded in learner history for trend analysis
5. Dataset retrieval is filtered to the detected level

## Model Limitations

- **Training domain**: Synthetic AI/ML Q&A pairs — may not generalize perfectly to other educational domains (e.g., biology, history)
- **Input length**: Optimized for questions ≤ 512 tokens (DistilBERT max)
- **Balance**: Fine-tuned on balanced dataset (800 per class) — may have bias on extremely rare difficulty distributions
- **Real-time feedback**: Trained on static question snapshots — does not adapt to individual learner history

## Citation

If you use this model in your research or project, please cite:

```bibtex
@misc{eduagent2024,
  author = {Suravajjula, Sneha},
  title = {EduAgent: Adaptive AI Tutor with Personalized Learning Loop},
  year = {2024},
  url = {https://github.com/Sneha-260805/EduAgent}
}
```

## License

This model is released under the same license as the EduAgent project. See the root repository [`LICENSE`](../../LICENSE) for details.

## HuggingFace Hub

Model weights are also published at:

**[`Sneha-260805/distilbert-eduagent-v2`](https://huggingface.co/Sneha-260805/distilbert-eduagent-v2)**

This allows direct loading via HuggingFace `transformers` without cloning the entire EduAgent repository.

---

**For full system architecture, training details, and experimental results, see [`report.tex`](../../report.tex).**
