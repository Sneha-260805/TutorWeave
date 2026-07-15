import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.multi_turn_evaluation import (
    _compute_behavior_metrics,
    _default_response_fn,
    build_learner_profiles,
    build_learning_sessions,
    run_multi_turn_evaluation,
    simulate_learner_turn,
)


def test_build_learner_profiles_and_run_small_session(tmp_path):
    profiles = build_learner_profiles()
    assert set(profiles.keys()) == {"beginner", "intermediate", "advanced"}

    results = run_multi_turn_evaluation(
        variants=["full", "plain_llm"],
        profiles=["beginner"],
        sessions=1,
        turns=3,
        output_dir=str(tmp_path),
        response_fn=lambda *args, **kwargs: {
            "answer": "A clear explanation with a short example.",
            "topic": "backpropagation",
            "level": "beginner",
            "teaching_mode": "default",
        },
    )

    sessions = build_learning_sessions(profiles=["beginner"], session_count=2)
    assert len(sessions) == 2
    assert all(len(session["turns"]) == 6 for session in sessions)

    assert len(results) == 2
    assert all("variant" in row for row in results)
    assert all("personalization" in row for row in results)
    assert all("learning_gain" in row for row in results)
    assert all("quiz_improvement" in row for row in results)
    assert os.path.exists(os.path.join(str(tmp_path), "multi_turn_statistical_tests.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "multi_turn_summary.csv"))
    assert os.path.exists(os.path.join(str(tmp_path), "multi_turn_session_statistics.csv"))


def test_simulate_learner_turn_updates_misconceptions_and_followups():
    state = {
        "profile_name": "beginner",
        "misconceptions": {"gradient descent": ["learning rate scheduling"]},
        "weak_concepts": ["learning rate scheduling"],
        "mastery": {"gradient descent": 0.35},
        "topic_counts": {"gradient descent": 1},
        "turn_history": [],
        "confidence": 0.25,
    }

    simulated = simulate_learner_turn(
        state=state,
        topic="gradient descent",
        tutor_answer="A short explanation that focuses on the learning rate and uses an example.",
        judge_scores={"personalization": 4.0, "adaptive_teaching": 4.0, "difficulty_alignment": 4.0, "weak_concept_recall": 4.0, "pedagogical_quality": 4.0, "hallucination_risk": 4.0},
        turn_idx=1,
        base_question="What does gradient descent do?",
    )

    assert simulated["should_follow_up"] is True
    assert simulated["misconception"] in simulated["follow_up_question"]
    assert simulated["confidence"] != state["confidence"]
    assert simulated["knowledge_state"] >= state.get("knowledge_state", 0.0)
    assert simulated["weak_concepts"]


def test_ablation_variants_produce_distinct_behavioral_signatures():
    profile = build_learner_profiles()["beginner"]
    question = "Why does gradient descent overshoot?"
    topic = "gradient descent"
    eval_outcome = {"knowledge_state": 0.6, "should_follow_up": True, "recommended_action": "re-explain"}
    state = {
        "profile_name": "beginner",
        "misconceptions": {topic: ["learning rate scheduling"]},
        "weak_concepts": ["learning rate scheduling"],
        "mastery": {topic: 0.35},
        "topic_counts": {topic: 2},
        "turn_history": [],
        "confidence": 0.25,
        "knowledge_state": 0.35,
    }

    full_answer = _default_response_fn(question, profile, "full", 1, topic, "beginner")["answer"]
    no_memory_answer = _default_response_fn(question, profile, "no_memory", 1, topic, "beginner")["answer"]
    no_evaluator_answer = _default_response_fn(question, profile, "no_evaluator", 1, topic, "beginner")["answer"]
    no_weak_retrieval_answer = _default_response_fn(question, profile, "no_weak_retrieval", 1, topic, "beginner")["answer"]
    no_retrieval_answer = _default_response_fn(question, profile, "no_retrieval", 1, topic, "beginner")["answer"]
    no_classifier_answer = _default_response_fn(question, profile, "no_classifier", 1, topic, "beginner")["answer"]
    plain_llm_answer = _default_response_fn(question, profile, "plain_llm", 1, topic, "beginner")["answer"]

    assert "Learner history" in full_answer
    assert "Strategy hint" in full_answer
    assert "Focus on weak concepts" in full_answer
    assert "retrieved example" in full_answer.lower()

    assert "Learner history" not in no_memory_answer
    assert "Strategy hint" not in no_evaluator_answer
    assert "Focus on weak concepts" not in no_weak_retrieval_answer
    assert "without retrieval grounding" in no_retrieval_answer.lower()
    assert "simple" in no_classifier_answer.lower() or "beginner" in no_classifier_answer.lower()
    assert "concise explanation" in plain_llm_answer.lower()

    full_metrics = _compute_behavior_metrics("beginner", topic, question, full_answer, state, eval_outcome)
    no_retrieval_metrics = _compute_behavior_metrics("beginner", topic, question, no_retrieval_answer, state, eval_outcome)
    no_weak_retrieval_metrics = _compute_behavior_metrics("beginner", topic, question, no_weak_retrieval_answer, state, eval_outcome)

    assert full_metrics["weak_concept_recall"] > no_weak_retrieval_metrics["weak_concept_recall"]
    assert no_retrieval_metrics["hallucination_rate"] > full_metrics["hallucination_rate"]
