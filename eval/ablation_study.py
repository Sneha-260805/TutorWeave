"""
EduAgent Ablation Study
========================
Quantifies the contribution of each pipeline component (difficulty classifier,
RAG retrieval, weak-area retrieval, memory hints, evaluator feedback loop) by
running a fixed test set through controlled variants of generate_tutor_response
and scoring outputs with an LLM-as-judge.

Variants:
    full                - complete pipeline (paper's proposed system)
    no_classifier       - level fixed to "intermediate" (classifier disabled)
    no_retrieval        - no RAG examples injected (primary + weak-area)
    no_weak_retrieval   - primary RAG kept, weak-area RAG disabled
    no_memory           - memory hint narrative disabled
    no_evaluator_loop   - teaching mode forced to "default" (no re-teach/advance)
    plain_llm           - raw LLM baseline, no scaffolding at all (Table-3 baseline)

Each variant is run against a fixed profile condition ("baseline" = empty
profile, "personalized" = simulated 8-session learner profile) so you can
separately report (a) the effect of each component and (b) the effect of
having learner history at all.

Usage:
    python -m eval.ablation_study
    python -m eval.ablation_study --n-questions 20 --repeats 3
    python -m eval.ablation_study --variants full,no_memory,plain_llm --profile-conditions personalized

Outputs (written to eval/):
    ablation_raw_results.csv     - one row per (variant, profile, question, run)
    ablation_summary.json        - aggregated means/stds/deltas/p-values
    ablation_report.md           - paper-ready markdown table + discussion
    ablation_chart.png           - bar chart of mean overall judge score per variant
"""

import argparse
import copy
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from statistics import mean, stdev

import numpy as np
from scipy.stats import wilcoxon
from bert_score import score as bert_score
from rouge_score import rouge_scorer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ml.classifier import predict_level
from ml.topic_detector import detect_best_topic
from ml.retriever import retrieve_examples, retrieve_for_weak_areas, df as RETRIEVER_DF
from agents.memory_agent import (
    build_memory_hint,
    build_evaluation_strategy_hint,
    ensure_profile_structure,
)
from agents.tutor_agent import (
    format_examples,
    infer_teaching_mode,
    build_mode_specific_instruction,
    build_local_fallback_answer,
)
from agents.llm_client import complete_chat
from config.settings import RAG_TOP_N, RAG_WEAK_TOP_N, MODEL_NAME, JUDGE_PROVIDER, JUDGE_MODEL

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))


def _normalize_question_item(item):
    if isinstance(item, dict):
        return {
            "question_id": item.get("question_id"),
            "difficulty": item.get("difficulty"),
            "components": item.get("components"),
            "question": item.get("question") if item.get("question") is not None else str(item),
        }
    return {
        "question_id": None,
        "difficulty": None,
        "components": None,
        "question": str(item),
    }


def load_question_bank(path: str) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Question file not found: {path}")
    df = pd.read_csv(path, dtype=str)
    if "question" not in df.columns:
        raise ValueError("Question file must contain a 'question' column.")

    def _maybe_str(value):
        return None if pd.isna(value) else str(value)

    bank = []
    for _, row in df.iterrows():
        bank.append({
            "question_id": _maybe_str(row.get("id")) if "id" in df.columns else None,
            "difficulty": _maybe_str(row.get("difficulty")) if "difficulty" in df.columns else None,
            "components": _maybe_str(row.get("components")) if "components" in df.columns else None,
            "question": str(row["question"]),
        })
    return bank

# ---------------------------------------------------------------------------
# Fixed test set — stratified across levels/topics.
# Extend this list (aim for 30-40) for the actual paper run; keep it a fixed,
# held-out set that was NOT used for classifier training or dataset generation.
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    # CNNs
    "What is a convolutional neural network and why is it useful for image data?",
    "How do convolution and pooling layers work together in a CNN?",
    "Why is batch normalization helpful in convolutional networks?",
    # RNNs
    "What is a recurrent neural network and how does it process sequence data?",
    "What problem do LSTMs and GRUs solve in RNN training?",
    "Describe how an RNN can be used for language modeling.",
    # Transformers and attention
    "Explain the attention mechanism in transformers.",
    "What is positional encoding and why is it needed in transformers?",
    "How do self-attention and cross-attention differ in encoder-decoder models?",
    # GANs
    "What are the main components of a GAN, and how do they compete?",
    "Why can GAN training be unstable, and what techniques help stabilize it?",
    "How is a conditional GAN different from a standard GAN?",
    # Reinforcement learning
    "What is the intuition behind reinforcement learning rewards?",
    "How does Q-learning differ from policy gradient methods?",
    "Explain the exploration-exploitation tradeoff in RL.",
    # Diffusion
    "What is a diffusion model and how is it used for image generation?",
    "How do forward and reverse diffusion processes work in generative models?",
    "Why are diffusion models often more stable than GANs for high-quality generation?",
    # PCA and SVM
    "What does PCA do and how can it help with dimensionality reduction?",
    "How do you choose the number of principal components to keep?",
    "What is the margin in a support vector machine?",
    "How does an SVM handle nonlinearly separable data?",
    # Bayesian learning
    "What is the difference between prior and posterior distributions?",
    "How does Bayesian learning update beliefs with new evidence?",
    "What is Bayesian model averaging and why is it useful?",
    # RAG, vector DB, embeddings, prompt engineering, fine-tuning
    "What is retrieval-augmented generation (RAG) and why use it?",
    "How does a vector database store and search embeddings?",
    "What is the role of embeddings in semantic search?",
    "Why is prompt engineering important for LLMs?",
    "What is the difference between prompt tuning and fine-tuning?",
    "How can you make a prompt more robust for a beginner learner?",
    # Core ML concepts and personalization
    "What is overfitting and how do you fix it?",
    "What is a convolutional neural network used for?",
    "How does the Adam optimizer improve on SGD?",
    "What is transfer learning and when should you use it?",
    "Why do we normalize input features before training?",
    "How does backpropagation work?",
    "What is the vanishing gradient problem?",
    "What is the difference between supervised and unsupervised learning?",
    "What is reinforcement learning in simple terms?",
    "What is the role of the softmax function?",
    "Explain the bias-variance tradeoff.",
    "What are the main components of a GAN?",
    "How does k-means clustering work?",
    "What is the intuition behind attention mechanisms?",
    "Why are embeddings useful for question answering systems?",
    "How can a tutor personalize explanations for a learner's background?",
    "What is the purpose of fine-tuning a language model on a domain-specific dataset?",
    "How do you use a vector DB with a retrieval model in a RAG pipeline?",
    "What is the role of memory and learner history in adaptive tutoring?",
]

PERSONALIZED_PROFILE = ensure_profile_structure({
    "sessions": 8,
    "questions_asked": 24,
    "last_level": "intermediate",
    "level_history": ["beginner", "beginner", "intermediate", "intermediate",
                       "intermediate", "advanced", "intermediate", "intermediate"],
    "topics_seen": ["gradient descent", "neural networks", "backpropagation",
                     "transformers", "overfitting", "cnn"],
    "topic_counts": {
        "gradient descent": 5, "backpropagation": 4, "transformers": 3,
        "overfitting": 2, "neural networks": 4, "cnn": 2,
    },
    "weak_areas": {
        "backpropagation": ["chain rule", "gradient flow"],
        "transformers": ["query-key-value mechanics"],
        "gradient descent": ["learning rate scheduling"],
    },
    "mastery": {
        "gradient descent": 0.72, "neural networks": 0.65, "backpropagation": 0.41,
        "transformers": 0.55, "overfitting": 0.78, "cnn": 0.60,
    },
    "used_explanations": {
        "gradient descent": ["beginner-default", "intermediate-default"],
        "backpropagation": ["beginner-default", "intermediate-remedial"],
        "transformers": ["intermediate-default"],
        "neural networks": ["beginner-default", "intermediate-advance"],
    },
    "last_evaluation": {
        "topic": "backpropagation", "understanding_level": "partial",
        "weak_concepts": ["chain rule"], "recommended_action": "re-explain",
    },
})

BASELINE_PROFILE = ensure_profile_structure({})

PROFILE_CONDITIONS = {
    "baseline": BASELINE_PROFILE,
    "personalized": PERSONALIZED_PROFILE,
}

# ---------------------------------------------------------------------------
# Variant -> component flags
# ---------------------------------------------------------------------------
VARIANT_FLAGS = {
    "full":              dict(classifier=True,  retrieval=True,  weak_retrieval=True,  memory=True,  evaluator_loop=True),
    "no_classifier":     dict(classifier=False, retrieval=True,  weak_retrieval=True,  memory=True,  evaluator_loop=True),
    "no_retrieval":      dict(classifier=True,  retrieval=False, weak_retrieval=False, memory=True,  evaluator_loop=True),
    "no_weak_retrieval": dict(classifier=True,  retrieval=True,  weak_retrieval=False, memory=True,  evaluator_loop=True),
    "no_memory":         dict(classifier=True,  retrieval=True,  weak_retrieval=True,  memory=False, evaluator_loop=True),
    "no_evaluator_loop": dict(classifier=True,  retrieval=True,  weak_retrieval=True,  memory=True,  evaluator_loop=False),
    "plain_llm":         dict(classifier=False, retrieval=False, weak_retrieval=False, memory=False, evaluator_loop=False),
}

DEFAULT_LEVEL_WHEN_NO_CLASSIFIER = "intermediate"


def generate_response(question: str, profile: dict, variant: str, temperature: float = 0.2):
    """
    Re-implements generate_tutor_response() from agents/tutor_agent.py with
    each component individually toggleable via VARIANT_FLAGS. Kept as a
    parallel function (rather than monkey-patching) so the exact ablation
    logic is auditable and citable in the paper's appendix.
    """
    flags = VARIANT_FLAGS[variant]

    if variant == "plain_llm":
        answer = complete_chat(
            [
                {"role": "system", "content": "You are a helpful AI/ML tutor. Answer the student's question clearly."},
                {"role": "user", "content": question},
            ],
            fallback="The tutor model is temporarily unavailable.",
            temperature=temperature,
            max_tokens=512,
        )
        return {
            "level": "n/a", "confidence": None, "topic": "n/a",
            "teaching_mode": "n/a", "answer": answer,
        }

    # --- classifier ---
    if flags["classifier"]:
        level, confidence = predict_level(question)
    else:
        level, confidence = DEFAULT_LEVEL_WHEN_NO_CLASSIFIER, None

    topic = detect_best_topic(question, level, RETRIEVER_DF) or "general"

    # --- retrieval ---
    if flags["retrieval"]:
        examples = retrieve_examples(question, level, top_n=RAG_TOP_N)
    else:
        examples = pd.DataFrame(columns=["question", "answer", "topic"])
    examples_text = format_examples(examples)

    weak_concepts = profile.get("weak_areas", {}).get(topic, [])
    if flags["retrieval"] and flags["weak_retrieval"]:
        already_retrieved = set(examples["question"].tolist()) if len(examples) > 0 else set()
        weak_examples = retrieve_for_weak_areas(
            weak_concepts, topic, level, top_n=RAG_WEAK_TOP_N, exclude_questions=already_retrieved
        )
    else:
        weak_examples = pd.DataFrame(columns=["question", "answer", "topic"])
    weak_examples_text = format_examples(weak_examples) if len(weak_examples) > 0 else ""

    # --- memory hint ---
    if flags["memory"]:
        memory_hint = build_memory_hint(profile, topic)
    else:
        memory_hint = ""

    # --- evaluator feedback loop (teaching mode) ---
    if flags["evaluator_loop"]:
        evaluation_strategy_hint = build_evaluation_strategy_hint(profile, topic)
        teaching_mode = infer_teaching_mode(evaluation_strategy_hint, profile, topic)
    else:
        evaluation_strategy_hint = ""
        teaching_mode = "default"
    mode_instruction = build_mode_specific_instruction(teaching_mode)

    weak_rag_section = (
        f"\nRetrieved examples targeting your weak areas ({', '.join(weak_concepts)}):\n{weak_examples_text}"
        if weak_examples_text else ""
    )
    memory_is_informative = memory_hint and "no strong prior history" not in memory_hint.lower()
    memory_section = f"\nLearner context: {memory_hint}" if memory_is_informative else ""
    eval_section = f"\nStrategy hint: {evaluation_strategy_hint}" if evaluation_strategy_hint else ""

    system_message = (
        f"You are EduAgent, an adaptive AI tutor specializing in {topic}. "
        f"You MUST stay focused on {topic} throughout your answer. "
        f"Do not drift into unrelated ML concepts unless {topic} itself explicitly requires them. "
        f"Pitch your explanation at {level} level."
    )
    user_message = f"""QUESTION: {question}
TOPIC: {topic}{memory_section}{eval_section}

RETRIEVED KNOWLEDGE BASE EXAMPLES — these are your PRIMARY source:
{examples_text}{weak_rag_section}

GROUNDING RULES (follow strictly):
1. Build your answer directly upon the facts, concepts, and explanations in the retrieved examples above.
2. Every key claim in your answer should trace back to the retrieved examples.
3. You may expand or re-explain retrieved content — but do NOT ignore it.
4. Do NOT copy retrieved text verbatim — synthesize and adapt in your own words.
5. Do NOT introduce facts or concepts that contradict the retrieved examples.
6. Do NOT include citation markers or bracketed source references such as [1], [2], or [3].

LEVEL GUIDE:
- beginner: simple words, analogies, no jargon
- intermediate: moderate detail, 1-2 key terms explained
- advanced: technical depth, assume background knowledge

ANSWER FORMAT:
- Use short bullet points.
- Start with the direct answer in the first bullet.
- Use 3 to 5 bullets total.
- Keep each bullet to 1 or 2 short sentences.
- Add a final "Next step:" bullet only when a useful follow-up topic naturally fits.
- Do not use numbered citations, footnotes, or source brackets.

{mode_instruction}
Answer clearly and concisely in the bullet format above."""

    fallback_answer = build_local_fallback_answer(question, level, topic, examples)
    answer = complete_chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        fallback=fallback_answer,
        temperature=temperature,
        max_tokens=512,
    )

    return {
        "level": level, "confidence": confidence, "topic": topic,
        "teaching_mode": teaching_mode, "answer": answer,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict, impartial grader for an AI tutoring system. "
    "Score the tutor's answer on the requested criteria. "
    "Return ONLY a single JSON object with the exact keys below and integer values 1-5. "
    "Do not include any explanation, commentary, or markdown formatting."
)

JUDGE_CRITERIA = [
    "correctness",
    "level_fit",
    "clarity",
    "specificity",
    "hallucination",
    "pedagogical_usefulness",
    "retrieval_grounding",
    "completeness",
    "personalization",
]

JUDGE_TEMPLATE = """Student question: {question}
Target learner level: {level}

Tutor's answer:
{answer}

Score the answer from 1 (poor) to 5 (excellent) on each criterion:
- correctness: is the explanation factually accurate?
- level_fit: is the complexity and vocabulary appropriate for a "{level}" learner?
- clarity: is the answer well-organized and easy to follow?
- specificity: does it avoid vague filler and give a concrete explanation?
- hallucination: does the answer avoid unsupported or made-up claims?
- pedagogical_usefulness: would this answer help a learner understand the concept?
- retrieval_grounding: does the answer stay grounded in relevant retrieved knowledge and evidence?
- completeness: does the answer cover the core aspects of the question without missing key points?
- personalization: does the answer adapt to the learner's level and likely needs?

Return only the JSON object below with exactly these keys and no other text:
{{"correctness": <int>, "level_fit": <int>, "clarity": <int>, "specificity": <int>, "hallucination": <int>, "pedagogical_usefulness": <int>, "retrieval_grounding": <int>, "completeness": <int>, "personalization": <int>}}

Do not output any analysis or markdown."""


def judge_answer(question: str, level: str, answer: str, max_retries: int = 2) -> dict:
    prompt = JUDGE_TEMPLATE.format(question=question, level=level, answer=answer)
    fallback_scores = {k: 3 for k in JUDGE_CRITERIA}
    fallback_scores["overall"] = 3.0
    for attempt in range(max_retries + 1):
        raw = complete_chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            fallback=json.dumps(fallback_scores),
            temperature=0.0,
            max_tokens=500,
            provider=JUDGE_PROVIDER,
            model=JUDGE_MODEL,
        )
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        json_match = re.search(r"\{(?:[^{}]|\{[^{}]*\})*\}", cleaned, flags=re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
        try:
            data = json.loads(cleaned)
            scores = {k: int(data[k]) for k in JUDGE_CRITERIA}
            scores["overall"] = round(mean(scores.values()), 3)
            return scores
        except Exception:
            # Fallback: parse human-readable criterion lines if JSON was not produced.
            parsed_scores = {}
            for crit in JUDGE_CRITERIA:
                pattern = rf"^\W*{crit}\W*.*?Score\s*[:=]\s*([1-5])"
                m = re.search(pattern, raw, flags=re.IGNORECASE | re.MULTILINE)
                if m:
                    parsed_scores[crit] = int(m.group(1))
            if len(parsed_scores) == len(JUDGE_CRITERIA):
                parsed_scores["overall"] = round(mean(parsed_scores.values()), 3)
                return parsed_scores
            if attempt == max_retries:
                return {**{k: None for k in JUDGE_CRITERIA}, "overall": None}
            time.sleep(1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _compute_text_metrics(answer: str, reference: str) -> tuple[float | None, float | None, float | None]:
    if not answer or not reference:
        return None, None, None
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_res = scorer.score(reference, answer)
        rouge_l_f1 = round(rouge_res["rougeL"].fmeasure, 4)
    except Exception:
        rouge_l_f1 = None

    try:
        P, R, F1 = bert_score([answer], [reference], lang="en", rescale_with_baseline=True)
        bert_score_f1 = round(float(F1[0]), 4)
    except Exception:
        bert_score_f1 = None

    try:
        from ml.embedder import embed_model, semantic_available
        if semantic_available and embed_model is not None:
            vectors = embed_model.encode([answer, reference], convert_to_numpy=True, show_progress_bar=False)
            sim = float(np.dot(vectors[0], vectors[1]) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]) + 1e-12))
            semantic_similarity = round(sim, 4)
        else:
            semantic_similarity = None
    except Exception:
        semantic_similarity = None

    return rouge_l_f1, bert_score_f1, semantic_similarity


def _paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float | None, float | None]:
    if len(a) < 2 or len(a) != len(b):
        return None, None
    try:
        stat, p_value = wilcoxon(a, b)
        return round(stat, 4), round(p_value, 6)
    except Exception:
        return None, None


def _reference_answer(question: str, level: str) -> str | None:
    try:
        reference_df = retrieve_examples(question, level, top_n=1)
        if len(reference_df) > 0:
            return str(reference_df.iloc[0]["answer"])
    except Exception:
        pass
    return None


def run_study(variants, profile_conditions, questions, repeats, temperature, judge=True, sleep_between=0.0):
    rows = []
    total = len(variants) * len(profile_conditions) * len(questions) * repeats
    done = 0

    for variant in variants:
        for profile_name in profile_conditions:
            profile = copy.deepcopy(PROFILE_CONDITIONS[profile_name])
            for q_idx, question_item in enumerate(questions):
                question_item = _normalize_question_item(question_item)
                question_text = question_item["question"]
                for run_idx in range(repeats):
                    done += 1
                    print(f"[{done}/{total}] variant={variant} profile={profile_name} q{q_idx+1} run{run_idx+1}", flush=True)
                    start_ts = time.perf_counter()
                    result = generate_response(question_text, profile, variant, temperature=temperature)
                    latency_s = round(time.perf_counter() - start_ts, 4)

                    ref_level = result["level"] if result["level"] != "n/a" else predict_level(question_text)[0]
                    reference = _reference_answer(question_text, ref_level)
                    rouge_l_f1, bert_score_f1, semantic_similarity = _compute_text_metrics(result["answer"], reference)

                    row = {
                        "variant": variant,
                        "profile_condition": profile_name,
                        "question_idx": q_idx,
                        "question_id": question_item.get("question_id"),
                        "difficulty": question_item.get("difficulty"),
                        "components": question_item.get("components"),
                        "question": question_text,
                        "run_idx": run_idx,
                        "level": result["level"],
                        "topic": result["topic"],
                        "teaching_mode": result["teaching_mode"],
                        "answer": result["answer"],
                        "answer_words": len(result["answer"].split()),
                        "rouge_l_f1": rouge_l_f1,
                        "bert_score_f1": bert_score_f1,
                        "semantic_similarity": semantic_similarity,
                        "response_latency_s": latency_s,
                    }

                    if judge:
                        scores = judge_answer(question_text, result["level"], result["answer"])
                        row.update({f"judge_{k}": v for k, v in scores.items()})

                    rows.append(row)
                    if sleep_between:
                        time.sleep(sleep_between)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistics: paired comparison of each variant against "full", same profile
# ---------------------------------------------------------------------------
def paired_ttest(a, b):
    """Manual paired t-test (no scipy dependency). a, b: lists of equal length."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    n = len(diffs)
    if n < 2:
        return None, None
    d_mean = mean(diffs)
    d_std = stdev(diffs) if n > 1 else 0.0
    if d_std == 0:
        return d_mean, (0.0 if d_mean == 0 else float("inf"))
    t_stat = d_mean / (d_std / math.sqrt(n))
    return d_mean, t_stat


def build_summary(df: pd.DataFrame) -> dict:
    summary = {"variants": {}, "comparisons_vs_full": {}}
    metric_cols = [c for c in df.columns if (c.startswith("judge_") and c != "judge_overall") or c in {
        "judge_overall", "answer_words", "rouge_l_f1", "bert_score_f1", "semantic_similarity", "response_latency_s"
    }]
    metric_cols = [c for c in metric_cols if c in df.columns]

    for (variant, profile_condition), g in df.groupby(["variant", "profile_condition"]):
        key = f"{variant}__{profile_condition}"
        summary["variants"][key] = {}
        for col in metric_cols:
            vals = [v for v in g[col].tolist() if v is not None]
            if not vals:
                continue
            summary["variants"][key][col] = {
                "mean": round(mean(vals), 3),
                "std": round(stdev(vals), 3) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }

    # Paired deltas vs "full" within the same profile_condition, matched on question_idx+run_idx
    for profile_condition in df["profile_condition"].unique():
        full_g = df[(df["variant"] == "full") & (df["profile_condition"] == profile_condition)]
        full_g = full_g.set_index(["question_idx", "run_idx"]).sort_index()
        for variant in df["variant"].unique():
            if variant == "full":
                continue
            var_g = df[(df["variant"] == variant) & (df["profile_condition"] == profile_condition)]
            if var_g.empty:
                continue
            var_g = var_g.set_index(["question_idx", "run_idx"]).sort_index()
            common_idx = full_g.index.intersection(var_g.index)
            if len(common_idx) == 0:
                continue
            a = full_g.loc[common_idx, "judge_overall"].tolist()
            b = var_g.loc[common_idx, "judge_overall"].tolist()
            d_mean, t_stat = paired_ttest(a, b)
            comp_key = f"full_vs_{variant}__{profile_condition}"
            summary["comparisons_vs_full"][comp_key] = {
                "mean_delta_overall_score": round(d_mean, 3) if d_mean is not None else None,
                "t_statistic": round(t_stat, 3) if isinstance(t_stat, float) and math.isfinite(t_stat) else t_stat,
                "n_paired": len(common_idx),
                "note": "positive delta = full system scores higher than ablated variant",
            }

    return summary


def build_markdown_report(df: pd.DataFrame, summary: dict) -> str:
    lines = ["# EduAgent Ablation Study", "", "## Mean scores by variant (1-5 scale, LLM-as-judge)", ""]
    lines.append("| Variant | Profile | Correctness | Level Fit | Clarity | Specificity | Hallucination | Pedagogical Usefulness | Retrieval Grounding | Completeness | Personalization | Overall | ROUGE-L | BERTScore | Semantic Sim | Latency (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, m in summary["variants"].items():
        variant, profile_condition = key.split("__")
        def g(col):
            return m.get(col, {}).get("mean", "-")
        lines.append(
            f"| {variant} | {profile_condition} | {g('judge_correctness')} | {g('judge_level_fit')} | "
            f"{g('judge_clarity')} | {g('judge_specificity')} | {g('judge_hallucination')} | {g('judge_pedagogical_usefulness')} | "
            f"{g('judge_retrieval_grounding')} | {g('judge_completeness')} | {g('judge_personalization')} | **{g('judge_overall')}** | "
            f"{g('rouge_l_f1')} | {g('bert_score_f1')} | {g('semantic_similarity')} | {g('response_latency_s')} |"
        )

    lines += ["", "## Component contribution (paired delta vs. full system)", "",
              "Positive delta = removing that component hurt the overall judge score.", "",
              "| Comparison | Mean Δ (overall) | t-statistic | n paired |", "|---|---|---|---|"]
    for key, c in summary["comparisons_vs_full"].items():
        lines.append(f"| {key} | {c['mean_delta_overall_score']} | {c['t_statistic']} | {c['n_paired']} |")

    lines += [
        "", "## How to read this for the paper",
        "- Each row under 'Component contribution' isolates one architectural piece "
        "(classifier, retrieval, weak-area retrieval, memory hint, evaluator feedback loop) "
        "by comparing the full system against a version with only that piece removed, "
        "on the *same* questions and profile condition.",
        "- Report `n_paired` alongside deltas — increase `--repeats` and `--n-questions` "
        "for a paper-grade sample (recommended: >=30 questions x 3 repeats = 90 paired samples per comparison).",
        "- `full (personalized)` vs `full (baseline)` isolates the effect of *having* learner "
        "history at all, independent of which component uses it.",
        "- `plain_llm` reproduces the qualitative Table-3 comparison in the report as a quantitative baseline.",
    ]
    return "\n".join(lines)


def maybe_plot(summary: dict, out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping chart. pip install matplotlib to enable.")
        return

    keys = list(summary["variants"].keys())
    vals = [summary["variants"][k].get("judge_overall", {}).get("mean") for k in keys]
    pairs = [(k, v) for k, v in zip(keys, vals) if v is not None]
    if not pairs:
        return
    pairs.sort(key=lambda x: x[1])
    labels, scores = zip(*pairs)

    plt.figure(figsize=(9, max(3, 0.4 * len(labels))))
    plt.barh(labels, scores, color="#4C72B0")
    plt.xlabel("Mean overall judge score (1-5)")
    plt.title("EduAgent Ablation Study — mean answer quality by variant")
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Chart written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EduAgent ablation study")
    parser.add_argument("--variants", type=str, default=",".join(VARIANT_FLAGS.keys()),
                         help="comma-separated list of variants to run")
    parser.add_argument("--profile-conditions", type=str, default="baseline,personalized",
                         help="comma-separated: baseline,personalized")
    parser.add_argument("--question-file", type=str, default=None,
                         help="CSV file containing evaluation questions; must include a 'question' column")
    parser.add_argument("--n-questions", type=int, default=len(TEST_QUESTIONS),
                         help="how many questions to use from the selected evaluation set (default: all)")
    parser.add_argument("--repeats", type=int, default=1,
                         help="repeats per question per variant (use 3 for the paper run)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge scoring (faster, for debugging)")
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to sleep between calls (rate-limit safety)")
    args = parser.parse_args()

    variants = args.variants.split(",")
    profile_conditions = args.profile_conditions.split(",")
    if args.question_file:
        questions = load_question_bank(args.question_file)[: args.n_questions]
    else:
        questions = [{"question": q} for q in TEST_QUESTIONS[: args.n_questions]]

    print(f"Running ablation study: {len(variants)} variants x {len(profile_conditions)} profile "
          f"conditions x {len(questions)} questions x {args.repeats} repeats "
          f"= {len(variants) * len(profile_conditions) * len(questions) * args.repeats} LLM tutor calls "
          f"({'+ judge calls' if not args.no_judge else 'no judge'}).\n")

    df = run_study(
        variants=variants,
        profile_conditions=profile_conditions,
        questions=questions,
        repeats=args.repeats,
        temperature=args.temperature,
        judge=not args.no_judge,
        sleep_between=args.sleep,
    )

    raw_path = os.path.join(EVAL_DIR, "ablation_raw_results.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw results written to: {raw_path}")

    if not args.no_judge:
        summary = build_summary(df)
        summary_path = os.path.join(EVAL_DIR, "ablation_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Summary written to: {summary_path}")

        report = build_markdown_report(df, summary)
        report_path = os.path.join(EVAL_DIR, "ablation_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report written to: {report_path}")

        maybe_plot(summary, os.path.join(EVAL_DIR, "ablation_chart.png"))
    else:
        print("Judge scoring skipped (--no-judge); only raw answers were saved.")


if __name__ == "__main__":
    main()
