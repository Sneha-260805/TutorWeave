"""
Fast metric computation for EduAgent evaluation outputs.

This script supports two evaluation modes:
1. Standard single-turn outputs with reference_answer/answer columns.
2. Multi-turn tutoring results with metrics such as personalization, adaptive teaching,
   weak concept recall, difficulty alignment, pedagogical quality, hallucination rate,
   learning gain, and latency.

Usage:
    python metrics_fast.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency path
    SentenceTransformer = None

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

if SentenceTransformer is not None:
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:  # pragma: no cover - fallback if model download fails
        model = None
else:
    model = None

HERE = Path(__file__).resolve().parent
ROOT = HERE
OUT = ROOT / "analysis" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def bleu(ref: str, pred: str) -> float:
    return float(sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth))


def rouge_l(ref: str, pred: str) -> float:
    return float(rouge.score(ref, pred)["rougeL"].fmeasure)


def exact(ref: str, pred: str) -> int:
    return int(str(ref).strip().lower() == str(pred).strip().lower())


def distinct(text: str):
    tok = str(text).split()
    if not tok:
        return 0.0, 0.0
    d1 = len(set(tok)) / len(tok)
    bigrams = list(zip(tok, tok[1:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return d1, d2


def _encode_texts(texts: Iterable[str]):
    if model is not None:
        return model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    vectorizer = TfidfVectorizer().fit(list(texts))
    return vectorizer.transform(list(texts))


def evaluate_text_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    refs = df["reference_answer"].astype(str).tolist()
    preds = df["answer"].astype(str).tolist()

    ref_emb = _encode_texts(refs)
    pred_emb = _encode_texts(preds)
    sem = cosine_similarity(ref_emb, pred_emb).diagonal()

    rows = []
    for i, (r, p) in enumerate(zip(refs, preds), start=1):
        d1, d2 = distinct(p)
        rows.append(
            {
                "BLEU": bleu(r, p),
                "ROUGE_L": rouge_l(r, p),
                "Semantic": float(sem[i - 1]),
                "ExactMatch": exact(r, p),
                "Distinct1": d1,
                "Distinct2": d2,
                "ResponseLength": len(p),
            }
        )

    out = df.copy()
    for k in rows[0]:
        out[k] = [x[k] for x in rows]
    return out


def summarize_text_metrics(df: pd.DataFrame, group_cols: Iterable[str], output_path: Path) -> pd.DataFrame:
    summary_cols = [
        "BLEU",
        "ROUGE_L",
        "Semantic",
        "ExactMatch",
        "Distinct1",
        "Distinct2",
        "ResponseLength",
    ]
    summary = df.groupby(list(group_cols), dropna=False)[summary_cols].mean().round(4).reset_index()
    summary.to_csv(output_path, index=False)
    return summary


def evaluate_multi_turn_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    metric_cols = [
        "personalization",
        "adaptive_teaching",
        "difficulty_alignment",
        "weak_concept_recall",
        "pedagogical_quality",
        "hallucination_rate",
        "learning_gain",
        "latency_ms",
    ]
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        raise ValueError(f"No supported multi-turn metric columns found in {path}")

    group_cols = [c for c in ["variant", "profile"] if c in df.columns]
    if not group_cols:
        group_cols = ["variant"]

    summary = df.groupby(group_cols, dropna=False)[available].mean().round(4).reset_index()
    return summary


def main() -> None:
    print("Computing EduAgent evaluation metrics...")

    # Standard outputs: general/personal evaluation results.
    general_candidates = [
        ROOT / "ablation" / "results_general" / "responses.csv",
        ROOT / "eval" / "ablation" / "results_general" / "responses.csv",
    ]
    personal_candidates = [
        ROOT / "ablation" / "results_personalization" / "responses.csv",
        ROOT / "eval" / "ablation" / "results_personalization" / "responses.csv",
    ]

    for path in general_candidates:
        if path.exists():
            general = pd.read_csv(path)
            if {"reference_answer", "answer"}.issubset(general.columns):
                gen_metrics = evaluate_text_dataframe(general, "General")
                gen_metrics.to_csv(OUT / "general_metrics_fast.csv", index=False)
                summarize_text_metrics(gen_metrics, ["variant"], OUT / "general_summary_fast.csv")
                print(f"Saved general metrics to {OUT / 'general_metrics_fast.csv'}")
            break

    for path in personal_candidates:
        if path.exists():
            personal = pd.read_csv(path)
            if {"reference_answer", "answer"}.issubset(personal.columns):
                per_metrics = evaluate_text_dataframe(personal, "Personalization")
                per_metrics.to_csv(OUT / "personal_metrics_fast.csv", index=False)
                summarize_text_metrics(per_metrics, ["variant"], OUT / "personal_summary_fast.csv")
                print(f"Saved personalization metrics to {OUT / 'personal_metrics_fast.csv'}")
            break

    multi_turn_path = ROOT / "eval" / "multi_turn_results.csv"
    if multi_turn_path.exists():
        summary = evaluate_multi_turn_results(multi_turn_path)
        summary.to_csv(OUT / "multi_turn_metrics_summary.csv", index=False)
        print(f"Saved multi-turn summary to {OUT / 'multi_turn_metrics_summary.csv'}")

    print("Done.")
    print(f"Files saved to: {OUT}")


if __name__ == "__main__":
    main()
