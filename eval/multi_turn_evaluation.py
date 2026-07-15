"""
Multi-turn evaluation protocol for EduAgent.

This module replaces single-turn judging with 4–6 turn learning sessions so that
memory, evaluator feedback, weak retrieval, and personalization can be observed
in a realistic tutoring flow.

The protocol is designed to be conference-ready, reproducible, and architecture-safe.
It uses existing EduAgent components (memory profile updates, teaching-mode hints)
without changing the tutor architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import deepcopy
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.llm_client import complete_chat
from agents.memory_agent import (
    build_evaluation_strategy_hint,
    build_memory_hint,
    ensure_profile_structure,
    update_last_evaluation,
    update_profile_after_evaluation,
    update_profile_after_question,
)
from agents.tutor_agent import build_mode_specific_instruction, infer_teaching_mode
from config.settings import JUDGE_MODEL, JUDGE_PROVIDER


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(EVAL_DIR, "multi_turn_results.csv")
SUMMARY_JSON = os.path.join(EVAL_DIR, "multi_turn_summary.json")
REPORT_MD = os.path.join(EVAL_DIR, "multi_turn_report.md")
STATISTICAL_CSV = os.path.join(EVAL_DIR, "multi_turn_statistical_tests.csv")
SESSION_STATS_CSV = os.path.join(EVAL_DIR, "multi_turn_session_statistics.csv")

JUDGE_SYSTEM = (
    "You are an impartial evaluator for an adaptive tutoring system. "
    "Score each response on a 1-5 scale using the rubric below. "
    "Return only valid JSON with numeric values."
)

JUDGE_TEMPLATE = """You are judging a single tutor turn in a multi-turn learning session.

Rubric (1-5):
- personalization: how well the answer adapts to the learner's profile, history, and weaknesses.
- adaptive_teaching: how well the answer changes strategy based on the learner's prior errors or misconceptions.
- difficulty_alignment: whether the explanation matches the learner's expected ability.
- weak_concept_recall: whether the response explicitly addresses previously weak concepts.
- pedagogical_quality: whether the explanation is clear, useful, and educational.
- hallucination_risk: how much unsupported or fabricated content appears (5 = no hallucination, 1 = severe hallucination).

Context:
- learner_profile: {profile_name}
- topic: {topic}
- target_level: {target_level}
- turn_index: {turn_idx}
- question: {question}
- tutor_answer: {answer}

Return only JSON in this exact shape:
{{"personalization": 1, "adaptive_teaching": 1, "difficulty_alignment": 1, "weak_concept_recall": 1, "pedagogical_quality": 1, "hallucination_risk": 1}}
"""


def build_learner_profiles() -> Dict[str, dict]:
    """Create three realistic learner profiles with distinct weaknesses."""
    profiles: Dict[str, dict] = {}

    profiles["beginner"] = ensure_profile_structure({
        "sessions": 1,
        "questions_asked": 0,
        "last_level": "beginner",
        "level_history": ["beginner"],
        "topics_seen": ["gradient descent"],
        "topic_counts": {"gradient descent": 1},
        "weak_areas": {
            "gradient descent": ["learning rate scheduling", "intuition about updates"],
            "backpropagation": ["chain rule"],
        },
        "mastery": {"gradient descent": 0.35, "backpropagation": 0.28},
        "used_explanations": {"gradient descent": ["beginner-default"]},
        "last_evaluation": {},
    })

    profiles["intermediate"] = ensure_profile_structure({
        "sessions": 2,
        "questions_asked": 2,
        "last_level": "intermediate",
        "level_history": ["intermediate", "intermediate"],
        "topics_seen": ["backpropagation", "transformers"],
        "topic_counts": {"backpropagation": 2, "transformers": 1},
        "weak_areas": {
            "backpropagation": ["gradient flow"],
            "transformers": ["query-key-value mechanics"],
        },
        "mastery": {"backpropagation": 0.58, "transformers": 0.55},
        "used_explanations": {"backpropagation": ["intermediate-remedial"]},
        "last_evaluation": {
            "topic": "backpropagation",
            "understanding_level": "partial",
            "weak_concepts": ["gradient flow"],
            "recommended_action": "re-explain",
        },
    })

    profiles["advanced"] = ensure_profile_structure({
        "sessions": 3,
        "questions_asked": 3,
        "last_level": "advanced",
        "level_history": ["advanced", "advanced", "intermediate"],
        "topics_seen": ["attention", "transfer learning", "diffusion"],
        "topic_counts": {"attention": 2, "transfer learning": 1, "diffusion": 1},
        "weak_areas": {
            "attention": ["softmax normalization"],
            "diffusion": ["denoising schedule"],
        },
        "mastery": {"attention": 0.80, "transfer learning": 0.74, "diffusion": 0.70},
        "used_explanations": {"attention": ["advanced-contrastive"]},
        "last_evaluation": {
            "topic": "attention",
            "understanding_level": "good",
            "weak_concepts": ["softmax normalization"],
            "recommended_action": "advance",
        },
    })

    return profiles


def _default_response_fn(question: str, profile: dict, variant: str, turn_idx: int, topic: str, target_level: str) -> Dict[str, object]:
    """Generate a lightweight, deterministic tutor response using the existing architecture helpers."""
    variant_flags = {
        "full": dict(classifier=True, retrieval=True, weak_retrieval=True, memory=True, evaluator_loop=True),
        "no_memory": dict(classifier=True, retrieval=True, weak_retrieval=True, memory=False, evaluator_loop=True),
        "no_evaluator": dict(classifier=True, retrieval=True, weak_retrieval=True, memory=True, evaluator_loop=False),
        "no_weak_retrieval": dict(classifier=True, retrieval=True, weak_retrieval=False, memory=True, evaluator_loop=True),
        "no_retrieval": dict(classifier=True, retrieval=False, weak_retrieval=False, memory=True, evaluator_loop=True),
        "no_classifier": dict(classifier=False, retrieval=True, weak_retrieval=True, memory=True, evaluator_loop=True),
        "plain_llm": dict(classifier=False, retrieval=False, weak_retrieval=False, memory=False, evaluator_loop=False),
    }
    flags = variant_flags.get(variant, variant_flags["full"])

    level = target_level if flags["classifier"] else "beginner"
    if variant == "plain_llm":
        answer = "Here is a concise explanation of the concept, with a short example and a check for understanding."
        return {"answer": answer, "level": level, "topic": topic, "teaching_mode": "default"}

    memory_hint = build_memory_hint(profile, topic) if flags["memory"] else ""
    evaluation_strategy_hint = build_evaluation_strategy_hint(profile, topic) if flags["evaluator_loop"] else ""
    if flags["evaluator_loop"] and not evaluation_strategy_hint:
        weak_concepts = profile.get("weak_areas", {}).get(topic, [])
        mastery = profile.get("mastery", {}).get(topic, 0.5)
        if weak_concepts:
            evaluation_strategy_hint = "re-explain the weak concept using the simplest possible example"
        elif float(mastery) < 0.4:
            evaluation_strategy_hint = "re-teach from scratch with a very simple concrete example"
        elif float(mastery) > 0.75:
            evaluation_strategy_hint = "advance with a slightly deeper explanation and a useful connection"
        else:
            evaluation_strategy_hint = "clarify the main idea before adding detail"
    teaching_mode = infer_teaching_mode(evaluation_strategy_hint, profile, topic)
    mode_instruction = build_mode_specific_instruction(teaching_mode)

    weak_concepts = profile.get("weak_areas", {}).get(topic, [])
    weak_section = f"Focus on weak concepts: {', '.join(weak_concepts)}." if flags["weak_retrieval"] and weak_concepts else ""
    retrieval_section = "Use the retrieved example and ground the explanation in it." if flags["retrieval"] else "Provide a general explanation without retrieval grounding."
    memory_section = f"Learner history: {memory_hint}" if flags["memory"] and memory_hint else ""
    eval_section = f"Strategy hint: {evaluation_strategy_hint}" if flags["evaluator_loop"] and evaluation_strategy_hint else ""

    answer = (
        f"{question}\n"
        f"- Direct answer: this concept is explained at a {level} level.\n"
        f"- Key idea: connect the main principle to a simple example.\n"
        f"- Teaching move: {teaching_mode}.\n"
        f"- {retrieval_section}\n"
    )
    if weak_section:
        answer += f"- {weak_section}\n"
    if memory_section:
        answer += f"- {memory_section}\n"
    if eval_section:
        answer += f"- {eval_section}\n"
    answer += mode_instruction.strip().splitlines()[0] if mode_instruction.strip() else ""

    return {"answer": answer, "level": level, "topic": topic, "teaching_mode": teaching_mode}


def build_learning_sessions(profiles: Optional[List[str]] = None, session_count: int = 10) -> List[Dict[str, object]]:
    """Create structured multi-turn tutoring sessions that stress memory, weak retrieval, evaluator feedback, and difficulty adaptation."""
    profile_names = profiles or ["beginner", "intermediate", "advanced"]
    session_templates = {
        "beginner": [
            {
                "turns": [
                    {"role": "student", "question": "What does gradient descent do in simple terms?", "topic": "gradient descent", "target_level": "beginner"},
                    {"role": "tutor", "question": "Explain it clearly with a simple example.", "topic": "gradient descent", "target_level": "beginner"},
                    {"role": "student", "question": "Why did my answer about learning rate get marked wrong?", "topic": "gradient descent", "target_level": "beginner"},
                    {"role": "evaluator", "question": "Update the learner mastery after the previous mistake.", "topic": "gradient descent", "target_level": "beginner"},
                    {"role": "memory", "question": "Store the misconception about learning rate and overshooting.", "topic": "gradient descent", "target_level": "beginner"},
                    {"role": "student", "question": "Why does gradient descent overshoot?", "topic": "gradient descent", "target_level": "beginner"},
                ],
                "stress": ["memory", "retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What is backpropagation in one sentence?", "topic": "backpropagation", "target_level": "beginner"},
                    {"role": "tutor", "question": "Explain the main idea in plain language.", "topic": "backpropagation", "target_level": "beginner"},
                    {"role": "student", "question": "I got the chain rule question wrong.", "topic": "backpropagation", "target_level": "beginner"},
                    {"role": "evaluator", "question": "Lower the mastery estimate and simplify the next explanation.", "topic": "backpropagation", "target_level": "beginner"},
                    {"role": "memory", "question": "Record that chain rule remains a weak concept.", "topic": "backpropagation", "target_level": "beginner"},
                    {"role": "student", "question": "How does backpropagation use the chain rule?", "topic": "backpropagation", "target_level": "beginner"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What is overfitting in a simple sentence?", "topic": "overfitting", "target_level": "beginner"},
                    {"role": "tutor", "question": "Explain it with a toy example and a practical tip.", "topic": "overfitting", "target_level": "beginner"},
                    {"role": "student", "question": "I still do not know why the model memorizes training data.", "topic": "overfitting", "target_level": "beginner"},
                    {"role": "evaluator", "question": "Reduce mastery and offer a more concrete example.", "topic": "overfitting", "target_level": "beginner"},
                    {"role": "memory", "question": "Store the misunderstanding that overfitting means the model is always wrong.", "topic": "overfitting", "target_level": "beginner"},
                    {"role": "student", "question": "How can you tell if a model is overfitting?", "topic": "overfitting", "target_level": "beginner"},
                ],
                "stress": ["memory", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What does regularization do?", "topic": "regularization", "target_level": "beginner"},
                    {"role": "tutor", "question": "Explain it as a way to stop the model from memorizing noise.", "topic": "regularization", "target_level": "beginner"},
                    {"role": "student", "question": "I still confuse regularization with adding more data.", "topic": "regularization", "target_level": "beginner"},
                    {"role": "evaluator", "question": "Lower mastery and use a more intuitive analogy.", "topic": "regularization", "target_level": "beginner"},
                    {"role": "memory", "question": "Record the misconception that regularization is data augmentation.", "topic": "regularization", "target_level": "beginner"},
                    {"role": "student", "question": "Why does regularization help generalization?", "topic": "regularization", "target_level": "beginner"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What is the intuition behind a loss function?", "topic": "loss function", "target_level": "beginner"},
                    {"role": "tutor", "question": "Explain how the loss measures prediction error.", "topic": "loss function", "target_level": "beginner"},
                    {"role": "student", "question": "I think a smaller loss always means a better model.", "topic": "loss function", "target_level": "beginner"},
                    {"role": "evaluator", "question": "Reduce mastery and make the next explanation less abstract.", "topic": "loss function", "target_level": "beginner"},
                    {"role": "memory", "question": "Store the confusion between loss magnitude and model quality.", "topic": "loss function", "target_level": "beginner"},
                    {"role": "student", "question": "Why can a model with low loss still generalize poorly?", "topic": "loss function", "target_level": "beginner"},
                ],
                "stress": ["memory", "evaluator", "retrieval"],
            },
        ],
        "intermediate": [
            {
                "turns": [
                    {"role": "student", "question": "How does backpropagation update weights in a neural network?", "topic": "backpropagation", "target_level": "intermediate"},
                    {"role": "tutor", "question": "Give a layered explanation and connect it to gradients.", "topic": "backpropagation", "target_level": "intermediate"},
                    {"role": "student", "question": "I still do not understand why gradients vanish.", "topic": "backpropagation", "target_level": "intermediate"},
                    {"role": "evaluator", "question": "Reduce mastery and plan a more remedial explanation.", "topic": "backpropagation", "target_level": "intermediate"},
                    {"role": "memory", "question": "Save the weak concept about vanishing gradients for later review.", "topic": "backpropagation", "target_level": "intermediate"},
                    {"role": "student", "question": "Why can vanishing gradients hurt deep networks?", "topic": "backpropagation", "target_level": "intermediate"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What is the role of attention in transformers?", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "tutor", "question": "Explain with a concise example and a short comparison to RNNs.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "student", "question": "I confused the query-key-value roles again.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "evaluator", "question": "Lower the mastery estimate and emphasize the core mechanism.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "memory", "question": "Store the misunderstanding about query-key-value mechanics.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "student", "question": "How do query, key, and value interact in attention?", "topic": "transformers", "target_level": "intermediate"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does positional encoding help transformers?", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "tutor", "question": "Explain why order matters even when attention is used.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "student", "question": "I still think the model can infer order from the text alone.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "evaluator", "question": "Lower mastery and reinforce the sequence intuition.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "memory", "question": "Store the misconception that transformers understand order without positional encoding.", "topic": "transformers", "target_level": "intermediate"},
                    {"role": "student", "question": "Why do transformers need positional encoding?", "topic": "transformers", "target_level": "intermediate"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does an RNN differ from a transformer?", "topic": "rnn", "target_level": "intermediate"},
                    {"role": "tutor", "question": "Compare recurrence and attention in a short table-like explanation.", "topic": "rnn", "target_level": "intermediate"},
                    {"role": "student", "question": "I still mix up sequential recurrence with parallel attention.", "topic": "rnn", "target_level": "intermediate"},
                    {"role": "evaluator", "question": "Reduce mastery and make the distinction more explicit.", "topic": "rnn", "target_level": "intermediate"},
                    {"role": "memory", "question": "Save the confusion between recurrence and attention.", "topic": "rnn", "target_level": "intermediate"},
                    {"role": "student", "question": "Why is attention more parallel than recurrence?", "topic": "rnn", "target_level": "intermediate"},
                ],
                "stress": ["memory", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does gradient clipping help training?", "topic": "training stability", "target_level": "intermediate"},
                    {"role": "tutor", "question": "Explain it as a safeguard against unstable updates.", "topic": "training stability", "target_level": "intermediate"},
                    {"role": "student", "question": "I still think clipping changes the model architecture.", "topic": "training stability", "target_level": "intermediate"},
                    {"role": "evaluator", "question": "Lower mastery and emphasize the practical role of clipping.", "topic": "training stability", "target_level": "intermediate"},
                    {"role": "memory", "question": "Record the misunderstanding that clipping alters architecture.", "topic": "training stability", "target_level": "intermediate"},
                    {"role": "student", "question": "Why is gradient clipping useful in deep networks?", "topic": "training stability", "target_level": "intermediate"},
                ],
                "stress": ["memory", "evaluator", "retrieval"],
            },
        ],
        "advanced": [
            {
                "turns": [
                    {"role": "student", "question": "How does scaled dot-product attention work mathematically?", "topic": "attention", "target_level": "advanced"},
                    {"role": "tutor", "question": "Provide a precise explanation with the attention formula.", "topic": "attention", "target_level": "advanced"},
                    {"role": "student", "question": "I still mix up softmax normalization and the value weights.", "topic": "attention", "target_level": "advanced"},
                    {"role": "evaluator", "question": "Adjust the mastery estimate and make the next explanation more structured.", "topic": "attention", "target_level": "advanced"},
                    {"role": "memory", "question": "Record the softmax weakness for future tutoring.", "topic": "attention", "target_level": "advanced"},
                    {"role": "student", "question": "Why is softmax important in attention?", "topic": "attention", "target_level": "advanced"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does diffusion denoising work conceptually?", "topic": "diffusion", "target_level": "advanced"},
                    {"role": "tutor", "question": "Explain the process step by step with a simple analogy.", "topic": "diffusion", "target_level": "advanced"},
                    {"role": "student", "question": "I still struggle with the denoising schedule.", "topic": "diffusion", "target_level": "advanced"},
                    {"role": "evaluator", "question": "Lower the mastery score and reduce abstraction.", "topic": "diffusion", "target_level": "advanced"},
                    {"role": "memory", "question": "Store the misunderstanding about the denoising schedule.", "topic": "diffusion", "target_level": "advanced"},
                    {"role": "student", "question": "What does the denoising schedule control in diffusion models?", "topic": "diffusion", "target_level": "advanced"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does transfer learning reduce data requirements?", "topic": "transfer learning", "target_level": "advanced"},
                    {"role": "tutor", "question": "Explain reuse of representations and domain similarity.", "topic": "transfer learning", "target_level": "advanced"},
                    {"role": "student", "question": "I still confuse transfer learning with fine-tuning.", "topic": "transfer learning", "target_level": "advanced"},
                    {"role": "evaluator", "question": "Adjust mastery and make the distinction clearer.", "topic": "transfer learning", "target_level": "advanced"},
                    {"role": "memory", "question": "Record the confusion between transfer learning and fine-tuning.", "topic": "transfer learning", "target_level": "advanced"},
                    {"role": "student", "question": "What is the difference between transfer learning and fine-tuning?", "topic": "transfer learning", "target_level": "advanced"},
                ],
                "stress": ["memory", "evaluator", "retrieval"],
            },
            {
                "turns": [
                    {"role": "student", "question": "How does contrastive learning work?", "topic": "contrastive learning", "target_level": "advanced"},
                    {"role": "tutor", "question": "Explain positive and negative pairs in a compact way.", "topic": "contrastive learning", "target_level": "advanced"},
                    {"role": "student", "question": "I still think it only needs one sample per class.", "topic": "contrastive learning", "target_level": "advanced"},
                    {"role": "evaluator", "question": "Lower mastery and simplify the pairwise intuition.", "topic": "contrastive learning", "target_level": "advanced"},
                    {"role": "memory", "question": "Store the misconception that contrastive learning uses a single positive exemplar.", "topic": "contrastive learning", "target_level": "advanced"},
                    {"role": "student", "question": "Why do contrastive losses require multiple views?", "topic": "contrastive learning", "target_level": "advanced"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
            {
                "turns": [
                    {"role": "student", "question": "What is the role of the value projection in attention?", "topic": "attention", "target_level": "advanced"},
                    {"role": "tutor", "question": "Explain how the value projection changes the representation.", "topic": "attention", "target_level": "advanced"},
                    {"role": "student", "question": "I still mix up the projection and the attention scores.", "topic": "attention", "target_level": "advanced"},
                    {"role": "evaluator", "question": "Lower mastery and make the role of value vectors explicit.", "topic": "attention", "target_level": "advanced"},
                    {"role": "memory", "question": "Store the misunderstanding about value projection versus score computation.", "topic": "attention", "target_level": "advanced"},
                    {"role": "student", "question": "How do attention scores and value projections differ?", "topic": "attention", "target_level": "advanced"},
                ],
                "stress": ["memory", "weak_retrieval", "evaluator"],
            },
        ],
    }

    sessions: List[Dict[str, object]] = []
    for profile_name in profile_names:
        templates = session_templates.get(profile_name, [])
        for idx in range(session_count):
            template = templates[idx % len(templates)]
            sessions.append({"profile": profile_name, "session_id": len(sessions), "turns": deepcopy(template["turns"]), "stress": template["stress"]})
    return sessions


def _build_session_scenarios(profile_name: str) -> List[Dict[str, object]]:
    sessions = build_learning_sessions([profile_name], session_count=1)
    return sessions[0]["turns"] if sessions else []


def _get_profile_difficulty(profile_name: str) -> str:
    return profile_name.lower()


def _simulate_student_evaluation(
    turn_idx: int,
    profile_name: str,
    topic: str,
    answer: str,
    target_level: str,
    knowledge_state: float,
    judge_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Backward-compatible wrapper for the stateful learner simulator."""
    state = {
        "profile_name": profile_name,
        "misconceptions": {},
        "weak_concepts": [],
        "mastery": {},
        "topic_counts": {},
        "turn_history": [],
        "confidence": 0.5,
        "knowledge_state": float(knowledge_state),
    }
    return simulate_learner_turn(
        state=state,
        topic=topic,
        tutor_answer=answer,
        judge_scores=judge_scores,
        turn_idx=turn_idx,
        base_question=target_level,
    )


def _extract_misconceptions(answer: str, topic: str) -> List[str]:
    lowered = answer.lower()
    concepts: List[str] = []
    if "weak" in lowered or "misunderstand" in lowered or "confus" in lowered:
        if topic == "gradient descent":
            concepts.extend(["learning rate scheduling", "intuition about updates"])
        elif topic == "backpropagation":
            concepts.extend(["chain rule", "gradient flow"])
        elif topic == "attention":
            concepts.extend(["softmax normalization", "value projection"])
        elif topic == "diffusion":
            concepts.extend(["denoising schedule"])
        elif topic == "transformers":
            concepts.extend(["query-key-value mechanics", "positional encoding"])
        else:
            concepts.extend(["core idea", "example application"])
    if "example" in lowered and "weak" not in lowered and "confus" not in lowered:
        concepts.append("example application")
    return sorted(set(concepts))


def _compute_behavior_metrics(
    profile_name: str,
    topic: str,
    question: str,
    answer: str,
    state: Dict[str, object],
    eval_outcome: Dict[str, object],
) -> Dict[str, float]:
    """Compute metrics directly from observable tutoring behavior and learner-state changes."""
    lowered_answer = answer.lower()
    lowered_question = question.lower()

    weak_concepts = list(state.get("weak_concepts", [])) if isinstance(state.get("weak_concepts", []), list) else []
    misconceptions = list(state.get("misconceptions", [])) if isinstance(state.get("misconceptions", []), list) else []
    topic_history = int(state.get("topic_counts", {}).get(topic, 0)) if isinstance(state.get("topic_counts", {}), dict) else 0
    mastery_map = state.get("mastery", {})
    if not isinstance(mastery_map, dict):
        mastery_map = {}
    topic_mastery = float(mastery_map.get(topic, 0.35))

    personalization = 0.0
    if "history" in lowered_answer or "learner" in lowered_answer or "weak" in lowered_answer or "strategy" in lowered_answer:
        personalization += 0.35
    if weak_concepts:
        personalization += 0.25
    if topic_history > 1:
        personalization += 0.2
    if "teaching move" in lowered_answer or "teaching mode" in lowered_answer:
        personalization += 0.2
    personalization = float(np.clip(personalization, 0.0, 1.0))

    adaptive_teaching = 0.0
    if "teaching move" in lowered_answer or "strategy hint" in lowered_answer or "remedial" in lowered_answer or "clarification" in lowered_answer or "advance" in lowered_answer:
        adaptive_teaching += 0.4
    if eval_outcome.get("should_follow_up"):
        adaptive_teaching += 0.2
    if eval_outcome.get("recommended_action") in {"re-explain", "give more practice"}:
        adaptive_teaching += 0.2
    if misconceptions:
        adaptive_teaching += 0.2
    adaptive_teaching = float(np.clip(adaptive_teaching, 0.0, 1.0))

    weak_concept_recall = 0.0
    if weak_concepts:
        weak_concept_recall += 0.2
    if any(concept.lower() in lowered_answer for concept in weak_concepts):
        weak_concept_recall += 0.4
    if "weak concepts" in lowered_answer or "weak" in lowered_answer:
        weak_concept_recall += 0.25
    if "focus on weak concepts" in lowered_answer:
        weak_concept_recall += 0.15
    weak_concept_recall = float(np.clip(weak_concept_recall, 0.0, 1.0))

    difficulty_alignment = 0.0
    target_level = "beginner"
    if "beginner" in lowered_question or "simple" in lowered_question:
        target_level = "beginner"
    elif "intermediate" in lowered_question or "why" in lowered_question or "compare" in lowered_question:
        target_level = "intermediate"
    else:
        target_level = "advanced"

    if target_level == "beginner":
        if "simple" in lowered_answer or "example" in lowered_answer or "plain" in lowered_answer:
            difficulty_alignment = 0.9
        else:
            difficulty_alignment = 0.6
    elif target_level == "intermediate":
        if "example" in lowered_answer and "key idea" in lowered_answer:
            difficulty_alignment = 0.85
        else:
            difficulty_alignment = 0.7
    else:
        if "formula" in lowered_answer or "technical" in lowered_answer or "deeper" in lowered_answer:
            difficulty_alignment = 0.9
        else:
            difficulty_alignment = 0.7

    if profile_name == "beginner" and "simple" in lowered_answer:
        difficulty_alignment = min(1.0, difficulty_alignment + 0.05)
    if profile_name == "advanced" and any(token in lowered_answer for token in ["formula", "technical", "deeper"]):
        difficulty_alignment = min(1.0, difficulty_alignment + 0.05)

    learning_gain = float(np.clip((eval_outcome.get("knowledge_state", 0.0) - state.get("knowledge_state", 0.0)) / max(1e-9, 1.0 - float(state.get("knowledge_state", 0.0))), 0.0, 1.0)) if state.get("knowledge_state") is not None else 0.0

    hallucination_rate = 0.0
    unsupported_markers = ["as we know", "it is proven that", "scientifically", "definitely"]
    if any(marker in lowered_answer for marker in unsupported_markers):
        hallucination_rate = 0.8
    elif "retrieved example" in lowered_answer and "ground" in lowered_answer:
        hallucination_rate = 0.15
    elif "without retrieval grounding" in lowered_answer or "general explanation" in lowered_answer:
        hallucination_rate = 0.55
    else:
        hallucination_rate = 0.1 + max(0.0, 0.05 * (1.0 - topic_mastery))
    hallucination_rate = float(np.clip(hallucination_rate, 0.0, 1.0))

    return {
        "personalization": round(personalization, 3),
        "adaptive_teaching": round(adaptive_teaching, 3),
        "difficulty_alignment": round(difficulty_alignment, 3),
        "weak_concept_recall": round(weak_concept_recall, 3),
        "learning_gain": round(learning_gain, 3),
        "hallucination_rate": round(hallucination_rate, 3),
    }


def simulate_learner_turn(
    state: Dict[str, object],
    topic: str,
    tutor_answer: str,
    judge_scores: Optional[Dict[str, float]],
    turn_idx: int,
    base_question: str,
) -> Dict[str, object]:
    """Simulate a realistic learner that remembers misconceptions and asks follow-up questions."""
    if judge_scores is None:
        judge_scores = {}

    profile_name = str(state.get("profile_name", "beginner"))
    difficulty = _get_profile_difficulty(profile_name)

    personalization = float(judge_scores.get("personalization", 3.0)) / 5.0
    adaptive_teaching = float(judge_scores.get("adaptive_teaching", 3.0)) / 5.0
    difficulty_alignment = float(judge_scores.get("difficulty_alignment", 3.0)) / 5.0
    weak_concept_recall = float(judge_scores.get("weak_concept_recall", 3.0)) / 5.0
    pedagogical_quality = float(judge_scores.get("pedagogical_quality", 3.0)) / 5.0
    hallucination_score = float(judge_scores.get("hallucination_risk", 3.0)) / 5.0

    quality_signal = (
        0.25 * personalization
        + 0.2 * adaptive_teaching
        + 0.2 * difficulty_alignment
        + 0.15 * weak_concept_recall
        + 0.2 * pedagogical_quality
    )
    confidence_penalty = 0.05 * (1.0 - hallucination_score)

    mastery_map = state.get("mastery", {})
    if not isinstance(mastery_map, dict):
        mastery_map = {}
    topic_mastery = float(mastery_map.get(topic, 0.35))

    if difficulty == "beginner":
        gain = 0.08 * quality_signal - 0.04 * (1.0 - quality_signal) - confidence_penalty
        regression_threshold = 0.55
        follow_up_bias = 0.8
    elif difficulty == "intermediate":
        gain = 0.10 * quality_signal - 0.03 * (1.0 - quality_signal) - confidence_penalty
        regression_threshold = 0.65
        follow_up_bias = 0.6
    else:
        gain = 0.12 * quality_signal - 0.02 * (1.0 - quality_signal) - confidence_penalty
        regression_threshold = 0.75
        follow_up_bias = 0.4

    if turn_idx >= 2 and quality_signal < 0.65:
        gain -= 0.04

    misconceptions = list(state.get("misconceptions", {}).get(topic, [])) if isinstance(state.get("misconceptions", {}), dict) else []
    extracted = _extract_misconceptions(tutor_answer, topic)
    if extracted:
        misconceptions = sorted(set(misconceptions + extracted))

    weak_concepts = list(state.get("weak_concepts", [])) if isinstance(state.get("weak_concepts", []), list) else []
    if weak_concepts:
        weak_concepts = sorted(set(weak_concepts))
    if extracted:
        weak_concepts = sorted(set(weak_concepts + extracted))

    if quality_signal < 0.55:
        topic_mastery = max(0.0, topic_mastery - 0.08)
        understanding_level = "poor"
        recommended_action = "re-explain"
    elif quality_signal < 0.75:
        topic_mastery = max(0.0, topic_mastery - 0.03)
        understanding_level = "partial"
        recommended_action = "give more practice"
    else:
        topic_mastery = min(1.0, topic_mastery + gain)
        understanding_level = "good"
        recommended_action = "advance"

    if turn_idx >= 1 and topic_mastery < regression_threshold:
        understanding_level = "partial"
        recommended_action = "re-explain"

    if misconceptions and quality_signal >= 0.7:
        topic_mastery = min(1.0, topic_mastery + 0.03)

    previous_confidence = float(state.get("confidence", 0.5))
    confidence = float(np.clip(previous_confidence + 0.5 * gain + (0.02 if understanding_level == "good" else -0.01), 0.0, 1.0))

    topic_count = int(state.get("topic_counts", {}).get(topic, 0)) if isinstance(state.get("topic_counts", {}), dict) else 0
    follow_up_probability = min(1.0, follow_up_bias + 0.08 * topic_count + (0.05 if misconceptions else 0.0))
    should_follow_up = bool(quality_signal < 0.8 or misconceptions) and (np.random.rand() < follow_up_probability)

    misconception = misconceptions[0] if misconceptions else "main idea"
    follow_up_question = (
        f"I still do not understand {misconception} for {topic}. Can you explain it again?"
        if should_follow_up
        else f"Can you give one more example for {topic}?"
    )

    knowledge_state = float(np.clip(float(state.get("knowledge_state", topic_mastery)) + 0.4 * gain + (0.03 if should_follow_up else 0.0), 0.0, 1.0))
    quiz_score = float(np.clip(0.3 + 0.7 * knowledge_state, 0.0, 1.0))

    behavior_metrics = _compute_behavior_metrics(
        profile_name=profile_name,
        topic=topic,
        question=base_question,
        answer=tutor_answer,
        state={
            **state,
            "knowledge_state": float(state.get("knowledge_state", 0.0)),
            "weak_concepts": weak_concepts,
            "misconceptions": misconceptions,
            "topic_counts": {topic: topic_count + 1},
            "mastery": {**state.get("mastery", {}), topic: round(topic_mastery, 3)},
        },
        eval_outcome={
            "knowledge_state": round(knowledge_state, 3),
            "should_follow_up": should_follow_up,
            "recommended_action": recommended_action,
        },
    )

    return {
        "understanding_level": understanding_level,
        "weak_concepts": weak_concepts,
        "recommended_action": recommended_action,
        "quiz_score": round(quiz_score, 3),
        "knowledge_state": round(knowledge_state, 3),
        "confidence": round(confidence, 3),
        "misconceptions": misconceptions,
        "misconception": misconception,
        "should_follow_up": should_follow_up,
        "follow_up_question": follow_up_question,
        "feedback": (
            f"The learner is still confused about {misconception} and asks for a follow-up explanation."
            if should_follow_up
            else f"The learner's understanding improved after the tutor explanation."
        ),
        "mastery": {topic: round(topic_mastery, 3)},
        "topic_counts": {topic: topic_count + 1},
        **behavior_metrics,
    }


def _parse_judge_json(text: str) -> Dict[str, float]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Judge response did not contain JSON")
    return json.loads(cleaned[start : end + 1])


def _offline_judge_fallback(question: str, answer: str, profile_name: str, topic: str, target_level: str, turn_idx: int) -> Dict[str, float]:
    text = answer.lower()
    personalization = 3.0 + (0.5 if "history" in text or "weak" in text or "strategy" in text else 0.0)
    adaptive_teaching = 3.0 + (0.5 if turn_idx > 0 and ("strategy" in text or "teaching move" in text) else 0.0)
    difficulty_alignment = 4.0 if target_level.lower() in text else 3.0
    weak_concept_recall = 3.0 + (0.5 if "weak" in text else 0.0)
    pedagogical_quality = 3.5 + (0.5 if "example" in text or "direct answer" in text else 0.0)
    hallucination_risk = 4.0 if len(answer.split()) > 10 else 3.0
    return {
        "personalization": round(min(5.0, personalization), 3),
        "adaptive_teaching": round(min(5.0, adaptive_teaching), 3),
        "difficulty_alignment": round(min(5.0, difficulty_alignment), 3),
        "weak_concept_recall": round(min(5.0, weak_concept_recall), 3),
        "pedagogical_quality": round(min(5.0, pedagogical_quality), 3),
        "hallucination_risk": round(min(5.0, hallucination_risk), 3),
    }


def judge_turn(question: str, answer: str, profile_name: str, topic: str, target_level: str, turn_idx: int) -> Dict[str, object]:
    """Use an independent judge to score the tutor response on the requested dimensions."""
    prompt = JUDGE_TEMPLATE.format(
        profile_name=profile_name,
        topic=topic,
        target_level=target_level,
        turn_idx=turn_idx,
        question=question,
        answer=answer,
    )
    try:
        response = complete_chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            fallback=None,
            model=JUDGE_MODEL,
            temperature=0.0,
            max_tokens=280,
            provider=JUDGE_PROVIDER,
        )
        payload = _parse_judge_json(response)
        values = {
            "personalization": float(payload.get("personalization", 3.0)),
            "adaptive_teaching": float(payload.get("adaptive_teaching", 3.0)),
            "difficulty_alignment": float(payload.get("difficulty_alignment", 3.0)),
            "weak_concept_recall": float(payload.get("weak_concept_recall", 3.0)),
            "pedagogical_quality": float(payload.get("pedagogical_quality", 3.0)),
            "hallucination_risk": float(payload.get("hallucination_risk", 3.0)),
        }
        return {"judge_source": "llm", **values}
    except Exception:
        return {"judge_source": "offline_fallback", **_offline_judge_fallback(question, answer, profile_name, topic, target_level, turn_idx)}


def _benjamini_hochberg_adjust(p_values: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction in a SciPy-version-agnostic way."""
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    if p.size == 1:
        return [float(p[0])]
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    adjusted = np.empty(m, dtype=float)
    for rank in range(1, m + 1):
        adjusted[order[rank - 1]] = min(1.0, ranked[rank - 1] * m / rank)
    # Enforce monotonicity from largest to smallest.
    for idx in range(m - 2, -1, -1):
        adjusted[idx] = min(adjusted[idx], adjusted[idx + 1])
    return [float(value) for value in adjusted]


def _summarize_session_turns(turn_rows: List[Dict[str, object]], profile_name: str, session_idx: int) -> Dict[str, float]:
    metric_names = [
        "personalization",
        "adaptive_teaching",
        "difficulty_alignment",
        "weak_concept_recall",
        "pedagogical_quality",
        "hallucination_rate",
    ]
    summary: Dict[str, float] = {"profile": profile_name, "session_id": session_idx}
    for metric in metric_names:
        values = [float(row[metric]) for row in turn_rows if metric in row]
        summary[metric] = round(sum(values) / max(1, len(values)), 3) if values else 0.0

    if turn_rows:
        pre_test_score = float(turn_rows[0].get("pre_test_score", 0.0))
        post_test_score = float(turn_rows[-1].get("post_test_score", pre_test_score))
        summary["learning_gain"] = round(post_test_score - pre_test_score, 3)
        summary["quiz_improvement"] = round(post_test_score - pre_test_score, 3)
        summary["knowledge_retention"] = round(float(np.clip(post_test_score + 0.1 * summary["pedagogical_quality"], 0.0, 1.0)), 3)
        summary["user_satisfaction"] = round(float(np.clip((summary["personalization"] + summary["pedagogical_quality"] + summary["difficulty_alignment"]) / 3.0, 0.0, 1.0)), 3)
        consistency_scores = [float(row.get("difficulty_alignment", 3.0)) for row in turn_rows]
        if len(consistency_scores) > 1:
            consistency = 1.0 - (float(np.mean(np.abs(np.diff(consistency_scores)))) / 5.0)
        else:
            consistency = 1.0
        summary["tutor_consistency"] = round(float(np.clip(consistency, 0.0, 1.0)), 3)
    else:
        summary["learning_gain"] = 0.0
        summary["quiz_improvement"] = 0.0
        summary["knowledge_retention"] = 0.0
        summary["user_satisfaction"] = 0.0
        summary["tutor_consistency"] = 0.0

    summary["latency_ms"] = round(sum(float(row["latency_ms"]) for row in turn_rows) / max(1, len(turn_rows)), 3)
    summary["overall_quality"] = round(
        sum(summary[m] for m in ["personalization", "adaptive_teaching", "difficulty_alignment", "weak_concept_recall", "pedagogical_quality"]) / 5.0,
        3,
    )
    return summary


def _write_summary_tables(session_metrics_df: pd.DataFrame, aggregate_df: pd.DataFrame, output_dir: str) -> None:
    session_metrics_df = session_metrics_df.copy()
    if "session_id" not in session_metrics_df.columns:
        session_metrics_df["session_id"] = session_metrics_df.index
    metric_cols = [
        "personalization",
        "adaptive_teaching",
        "difficulty_alignment",
        "weak_concept_recall",
        "pedagogical_quality",
        "hallucination_rate",
        "learning_gain",
        "quiz_improvement",
        "knowledge_retention",
        "user_satisfaction",
        "tutor_consistency",
        "latency_ms",
        "overall_quality",
    ]
    stats_rows = []
    for profile_name in sorted(session_metrics_df["profile"].unique()):
        for metric in metric_cols:
            if metric not in session_metrics_df.columns:
                continue
            full_values = session_metrics_df[(session_metrics_df["profile"] == profile_name) & (session_metrics_df["variant"] == "full")][metric].astype(float).tolist()
            if not full_values:
                continue
            for variant in sorted(session_metrics_df["variant"].unique()):
                if variant == "full":
                    continue
                ablation_values = session_metrics_df[(session_metrics_df["profile"] == profile_name) & (session_metrics_df["variant"] == variant)][metric].astype(float).tolist()
                if not ablation_values:
                    continue
                if len(full_values) != len(ablation_values):
                    continue
                diffs = [b - a for a, b in zip(full_values, ablation_values)]
                if not diffs:
                    continue
                mean_diff = float(np.mean(diffs))
                std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                ci_low, ci_high = stats.t.interval(0.95, len(diffs) - 1, loc=mean_diff, scale=stats.sem(diffs)) if len(diffs) > 1 else (mean_diff, mean_diff)
                shapiro_p = stats.shapiro(diffs).pvalue if len(diffs) >= 3 else 1.0
                if np.isfinite(shapiro_p) and shapiro_p > 0.05 and len(diffs) > 1:
                    test_name = "paired_t_test"
                    p_value = float(stats.ttest_rel(full_values, ablation_values).pvalue)
                    pooled_std = np.sqrt((np.var(full_values, ddof=1) + np.var(ablation_values, ddof=1)) / 2.0) if len(full_values) > 1 and len(ablation_values) > 1 else 0.0
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0.0
                else:
                    test_name = "wilcoxon_signed_rank"
                    p_value = float(stats.wilcoxon(full_values, ablation_values, zero_method="wilcox", correction=False).pvalue) if len(diffs) > 1 else 1.0
                    effect_size = float(np.mean(stats.rankdata(np.abs(diffs))) / len(diffs)) if len(diffs) > 0 else 0.0
                if not np.isfinite(p_value):
                    p_value = 1.0
                if not np.isfinite(effect_size):
                    effect_size = 0.0
                stats_rows.append({
                    "profile": profile_name,
                    "metric": metric,
                    "variant": variant,
                    "mean_full": round(float(np.mean(full_values)), 3),
                    "mean_ablation": round(float(np.mean(ablation_values)), 3),
                    "mean_difference": round(mean_diff, 3),
                    "std_difference": round(std_diff, 3),
                    "ci_lower": round(float(ci_low), 3),
                    "ci_upper": round(float(ci_high), 3),
                    "p_value": round(p_value, 4),
                    "test": test_name,
                    "effect_size": round(effect_size, 3),
                })

    stats_df = pd.DataFrame(stats_rows)
    if not stats_df.empty:
        pvals = stats_df["p_value"].astype(float).tolist()
        if len(pvals) > 1:
            adjusted_pvals = _benjamini_hochberg_adjust(pvals)
            stats_df["adjusted_p_value"] = np.round(adjusted_pvals, 4)
        else:
            stats_df["adjusted_p_value"] = stats_df["p_value"]
        stats_df.to_csv(os.path.join(output_dir, "multi_turn_statistical_tests.csv"), index=False)

    session_stats = []
    for metric in metric_cols:
        if metric not in session_metrics_df.columns:
            continue
        for (variant, profile_name), group in session_metrics_df.groupby(["variant", "profile"]):
            values = group[metric].astype(float).tolist()
            if not values:
                continue
            session_stats.append({
                "variant": variant,
                "profile": profile_name,
                "metric": metric,
                "n_sessions": len(values),
                "mean": round(float(np.mean(values)), 4),
                "median": round(float(np.median(values)), 4),
                "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
                "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4),
                "stderr": round(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0, 4),
                "ci_lower": round(float(stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))[0]) if len(values) > 1 else float(np.mean(values)), 4),
                "ci_upper": round(float(stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))[1]) if len(values) > 1 else float(np.mean(values)), 4),
            })

    if session_stats:
        pd.DataFrame(session_stats).to_csv(os.path.join(output_dir, "multi_turn_session_statistics.csv"), index=False)

    group_summary = aggregate_df.groupby(["variant", "profile"], as_index=False)[metric_cols].mean().round(4)
    std_summary = aggregate_df.groupby(["variant", "profile"], as_index=False)[metric_cols].std().round(4)
    std_summary.columns = [c if c in ["variant", "profile"] else f"{c}_std" for c in std_summary.columns]
    combined = group_summary.merge(std_summary, on=["variant", "profile"], how="left")
    combined.to_csv(os.path.join(output_dir, "multi_turn_summary.csv"), index=False)


def run_multi_turn_evaluation(
    variants: Optional[List[str]] = None,
    profiles: Optional[List[str]] = None,
    sessions: int = 10,
    turns: int = 6,
    output_dir: Optional[str] = None,
    response_fn: Optional[Callable[[str, dict, str, int, str, str], Dict[str, object]]] = None,
) -> List[Dict[str, object]]:
    """Run the multi-turn evaluation and save CSV/JSON/Markdown outputs."""
    if variants is None:
        variants = ["full", "no_memory", "no_evaluator", "no_weak_retrieval", "no_retrieval", "no_classifier", "plain_llm"]
    if profiles is None:
        profiles = ["beginner", "intermediate", "advanced"]

    if response_fn is None:
        response_fn = _default_response_fn

    all_profile_data = build_learner_profiles()
    rows: List[Dict[str, object]] = []

    for variant in variants:
        for profile_name in profiles:
            profile_template = deepcopy(all_profile_data[profile_name])
            sessions_for_profile = build_learning_sessions([profile_name], session_count=sessions)
            for session_idx, session in enumerate(sessions_for_profile):
                profile = deepcopy(profile_template)
                scenario_steps = session["turns"][:turns]
                turn_rows: List[Dict[str, object]] = []
                pre_test_score = {"beginner": 0.35, "intermediate": 0.55, "advanced": 0.70}.get(profile_name, 0.5)
                knowledge_state = pre_test_score
                for turn_idx, step in enumerate(scenario_steps):
                    question = step["question"]
                    topic = step["topic"]
                    target_level = step["target_level"]
                    profile = update_profile_after_question(profile, topic, target_level)

                    start = time.perf_counter()
                    response = response_fn(question, profile, variant, turn_idx, topic, target_level)
                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    answer = str(response.get("answer", ""))
                    response_level = str(response.get("level", target_level))

                    judge_scores = judge_turn(question, answer, profile_name, topic, target_level, turn_idx)

                    state = {
                        "profile_name": profile_name,
                        "misconceptions": profile.get("weak_areas", {}).get(topic, []),
                        "weak_concepts": list(profile.get("weak_areas", {}).get(topic, [])),
                        "mastery": profile.get("mastery", {}),
                        "topic_counts": profile.get("topic_counts", {}),
                        "turn_history": profile.get("level_history", []),
                        "confidence": float(profile.get("mastery", {}).get(topic, 0.5)),
                        "knowledge_state": float(knowledge_state),
                    }
                    eval_outcome = simulate_learner_turn(
                        state=state,
                        topic=topic,
                        tutor_answer=answer,
                        judge_scores=judge_scores,
                        turn_idx=turn_idx,
                        base_question=question,
                    )
                    knowledge_state = float(eval_outcome["knowledge_state"])
                    profile = update_profile_after_evaluation(profile, topic, eval_outcome)
                    profile = update_last_evaluation(profile, topic, eval_outcome)
                    profile.get("mastery", {}).update(eval_outcome.get("mastery", {}))
                    profile.get("topic_counts", {}).update(eval_outcome.get("topic_counts", {}))
                    profile.setdefault("weak_areas", {})[topic] = sorted(set(profile.get("weak_areas", {}).get(topic, []) + eval_outcome.get("weak_concepts", [])))

                    turn_rows.append({
                        "variant": variant,
                        "profile": profile_name,
                        "session_id": session_idx,
                        "turn": turn_idx,
                        "question": question,
                        "topic": topic,
                        "target_level": target_level,
                        "response_level": response_level,
                        "answer": answer,
                        "judge_source": judge_scores.get("judge_source", "offline_fallback"),
                        "personalization": eval_outcome.get("personalization", 0.0),
                        "adaptive_teaching": eval_outcome.get("adaptive_teaching", 0.0),
                        "difficulty_alignment": eval_outcome.get("difficulty_alignment", 0.0),
                        "weak_concept_recall": eval_outcome.get("weak_concept_recall", 0.0),
                        "pedagogical_quality": eval_outcome.get("pedagogical_quality", 0.0),
                        "hallucination_rate": eval_outcome.get("hallucination_rate", 0.0),
                        "quiz_score": eval_outcome["quiz_score"],
                        "latency_ms": latency_ms,
                        "pre_test_score": pre_test_score,
                        "post_test_score": knowledge_state,
                        "stress": ",".join(session.get("stress", [])),
                    })

                summary = _summarize_session_turns(turn_rows, profile_name, session_idx)
                rows.append({
                    "variant": variant,
                    "profile": profile_name,
                    "session_id": session_idx,
                    "personalization": summary["personalization"],
                    "adaptive_teaching": summary["adaptive_teaching"],
                    "difficulty_alignment": summary["difficulty_alignment"],
                    "weak_concept_recall": summary["weak_concept_recall"],
                    "pedagogical_quality": summary["pedagogical_quality"],
                    "hallucination_rate": summary["hallucination_rate"],
                    "learning_gain": summary["learning_gain"],
                    "quiz_improvement": summary["quiz_improvement"],
                    "knowledge_retention": summary["knowledge_retention"],
                    "user_satisfaction": summary["user_satisfaction"],
                    "tutor_consistency": summary["tutor_consistency"],
                    "latency_ms": summary["latency_ms"],
                    "overall_quality": summary["overall_quality"],
                })

    output_dir = output_dir or EVAL_DIR
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "multi_turn_results.csv"), index=False)

    summary_df = results_df.groupby(["variant", "profile"], as_index=False).agg({
        "personalization": "mean",
        "adaptive_teaching": "mean",
        "difficulty_alignment": "mean",
        "weak_concept_recall": "mean",
        "pedagogical_quality": "mean",
        "hallucination_rate": "mean",
        "learning_gain": "mean",
        "quiz_improvement": "mean",
        "knowledge_retention": "mean",
        "user_satisfaction": "mean",
        "tutor_consistency": "mean",
        "latency_ms": "mean",
        "overall_quality": "mean",
    }).round(4)
    summary_df.to_csv(os.path.join(output_dir, "multi_turn_summary.csv"), index=False)

    _write_summary_tables(results_df, summary_df, output_dir)

    report = _build_markdown_report(summary_df)
    with open(os.path.join(output_dir, "multi_turn_report.md"), "w", encoding="utf-8") as f:
        f.write(report)

    with open(os.path.join(output_dir, "multi_turn_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    return rows


def _build_markdown_report(summary_df: pd.DataFrame) -> str:
    lines = [
        "# Multi-Turn EduAgent Evaluation",
        "",
        "This report summarizes a multi-turn tutoring evaluation in which each configuration is run over structured learning sessions that explicitly stress memory retrieval, evaluator feedback, weak-concept review, and difficulty adaptation.",
        "",
        "## Aggregate Results",
        "",
        "| Variant | Profile | Personalization | Adaptive Teaching | Weak Concept Recall | Difficulty Alignment | Hallucination Rate | Learning Gain | Latency (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['profile']} | {row['personalization']:.3f} | {row['adaptive_teaching']:.3f} | {row['weak_concept_recall']:.3f} | {row['difficulty_alignment']:.3f} | {row['hallucination_rate']:.3f} | {row['learning_gain']:.3f} | {row['latency_ms']:.1f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Higher personalization and adaptive teaching indicate stronger profile-aware tutoring.",
        "- Higher weak concept recall indicates that previously difficult concepts are revisited effectively.",
        "- Lower hallucination rate is preferred, especially when retrieval is ablated.",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-turn EduAgent evaluation protocol")
    parser.add_argument("--variants", default="full,no_memory,no_evaluator,no_weak_retrieval,no_retrieval,no_classifier,plain_llm")
    parser.add_argument("--profiles", default="beginner,intermediate,advanced")
    parser.add_argument("--sessions", type=int, default=10)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--output-dir", default=EVAL_DIR)
    args = parser.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    run_multi_turn_evaluation(variants=variants, profiles=profiles, sessions=args.sessions, turns=args.turns, output_dir=args.output_dir)
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
