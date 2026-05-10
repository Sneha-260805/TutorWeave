import argparse
import inspect
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# =========================================================
# LABEL CONFIG
# =========================================================
LABELS = ["beginner", "intermediate", "advanced"]

LABEL2ID = {
    "beginner": 0,
    "intermediate": 1,
    "advanced": 2,
}

ID2LABEL = {
    0: "beginner",
    1: "intermediate",
    2: "advanced",
}


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# PYTORCH DATASET CLASS
# =========================================================
class DifficultyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.encodings = self.tokenizer(
            self.texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =========================================================
# DATA LOADING + CLEANING
# =========================================================
def load_and_prepare_dataset(data_path: str):
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    required_cols = {"question", "level"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Dataset must contain columns: {required_cols}. "
            f"Missing columns: {missing_cols}"
        )

    # Keep only the columns needed for classifier training
    df = df[["question", "level"]].copy()

    # Clean text
    df["question"] = df["question"].astype(str).str.strip()
    df["level"] = df["level"].astype(str).str.strip().str.lower()

    # Remove empty questions
    df = df[df["question"].str.len() > 0]

    # Keep only valid difficulty labels
    df = df[df["level"].isin(LABEL2ID.keys())]

    # Remove exact duplicate questions
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    # Convert labels into numeric IDs
    df["label_id"] = df["level"].map(LABEL2ID)

    print("\nDataset loaded successfully.")
    print(f"Total rows after cleaning: {len(df)}")

    print("\nClass distribution:")
    print(df["level"].value_counts())

    if len(df) < 50:
        raise ValueError("Dataset is too small after cleaning.")

    return df


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================
def create_splits(df, seed: int = 42):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df["label_id"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label_id"],
    )

    print("\nSplit sizes:")
    print(f"Train: {len(train_df)}")
    print(f"Validation: {len(val_df)}")
    print(f"Test: {len(test_df)}")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


# =========================================================
# METRICS
# =========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


# =========================================================
# TRAINING ARGUMENTS HELPER
# Supports both old and new versions of transformers:
# some use evaluation_strategy, newer versions use eval_strategy.
# =========================================================
def build_training_args(args):
    training_args_kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_dir": os.path.join(args.output_dir, "logs"),
        "logging_steps": 20,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
        "fp16": torch.cuda.is_available(),
    }

    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in params:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**training_args_kwargs)


# =========================================================
# SAVE TEST REPORTS
# =========================================================
def save_test_reports(trainer, test_dataset, test_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    y_true = predictions.label_ids
    y_pred = np.argmax(logits, axis=1)

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    print("\nFinal Test Classification Report:")
    print(report_text)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    with open(output_dir / "confusion_matrix.json", "w", encoding="utf-8") as f:
        json.dump(cm.tolist(), f, indent=2)

    predictions_df = test_df.copy()

    predictions_df["predicted_label_id"] = y_pred
    predictions_df["predicted_level"] = [ID2LABEL[int(i)] for i in y_pred]
    predictions_df["true_level"] = [ID2LABEL[int(i)] for i in y_true]
    predictions_df["correct"] = (
        predictions_df["predicted_label_id"] == predictions_df["label_id"]
    )

    predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)

    print(f"\nReports saved to: {output_dir}")


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================
def train(args):
    set_seed(args.seed)

    print("\nEduAgent DistilBERT Difficulty Classifier Training")
    print("--------------------------------------------------")
    print(f"Dataset path: {args.data_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_length}")
    print(f"Learning rate: {args.learning_rate}")
    print("--------------------------------------------------")

    # Load and clean dataset
    df = load_and_prepare_dataset(args.data_path)

    # Split dataset
    train_df, val_df, test_df = create_splits(df, seed=args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Load DistilBERT model
    model = DistilBertForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Prepare PyTorch datasets
    train_dataset = DifficultyDataset(
        texts=train_df["question"].tolist(),
        labels=train_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    val_dataset = DifficultyDataset(
        texts=val_df["question"].tolist(),
        labels=val_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    test_dataset = DifficultyDataset(
        texts=test_df["question"].tolist(),
        labels=test_df["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )

    print("\nTraining started...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_dataset)

    print("\nTest metrics:")
    print(test_metrics)

    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save label mapping
    label_info = {
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }

    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2)

    # Save training config
    config_info = vars(args)

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_info, f, indent=2)

    # Save dataset splits for reproducibility
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)

    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)

    # Save reports
    save_test_reports(
        trainer=trainer,
        test_dataset=test_dataset,
        test_df=test_df,
        output_dir=output_dir / "reports",
    )

    print("\nTraining complete.")
    print(f"Model saved to: {output_dir}")


# =========================================================
# CLI ARGUMENTS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT difficulty classifier for EduAgent."
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/eduagent_dataset_easy_v2.csv",
        help="Path to the EduAgent dataset CSV.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/distilbert_eduagent_v2",
        help="Directory to save the trained model.",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="distilbert-base-uncased",
        help="Base Hugging Face model.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size per device.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate.",
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay.",
    )

    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Warmup ratio.",
    )

    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=2,
        help="Early stopping patience.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)