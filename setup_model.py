"""
Quick script to download and save DistilBERT model with proper weights.
"""
from transformers import DistilBertForSequenceClassification, AutoTokenizer
import json
from pathlib import Path

output_dir = Path("models/distilbert_eduagent_v2")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading DistilBERT base model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label={0: "beginner", 1: "intermediate", 2: "advanced"},
    label2id={"beginner": 0, "intermediate": 1, "advanced": 2},
)

print("Saving model and tokenizer...")
model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

# Save label mapping
label_info = {
    "labels": ["beginner", "intermediate", "advanced"],
    "label2id": {"beginner": 0, "intermediate": 1, "advanced": 2},
    "id2label": {"0": "beginner", "1": "intermediate", "2": "advanced"},
}
with open(output_dir / "label_map.json", "w") as f:
    json.dump(label_info, f, indent=2)

print(f"Model saved to {output_dir}")
