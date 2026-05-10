"""
Personalization Evaluation Script
===================================
Runs 10 ML/AI questions through EduAgent twice:
  1. Baseline  — empty learner profile (no prior history)
  2. Personalized — simulated experienced profile with mastery, weak areas,
     and previously used explanation styles populated.

Writes eval/evaluation_results.md with a side-by-side comparison table.

Usage:
    python -m eval.run_evaluation
    # or
    python eval/run_evaluation.py
"""

import sys
import os
import textwrap
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tutor_agent import generate_tutor_response
from agents.memory_agent import ensure_profile_structure

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "evaluation_results.md")

# ---------------------------------------------------------------------------
# 10 test questions spanning beginner → advanced across multiple topics
# ---------------------------------------------------------------------------
TEST_QUESTIONS = [
    "What is gradient descent?",
    "How does a neural network learn?",
    "What is the vanishing gradient problem?",
    "Explain the attention mechanism in transformers.",
    "What is overfitting and how do you fix it?",
    "How does backpropagation work?",
    "What is the difference between supervised and unsupervised learning?",
    "What is a convolutional neural network used for?",
    "How does the Adam optimizer improve on SGD?",
    "What is transfer learning and when should you use it?",
]

# ---------------------------------------------------------------------------
# Simulated experienced learner profile
# ---------------------------------------------------------------------------
PERSONALIZED_PROFILE = ensure_profile_structure({
    "sessions": 8,
    "questions_asked": 24,
    "last_level": "intermediate",
    "level_history": ["beginner", "beginner", "intermediate", "intermediate",
                      "intermediate", "advanced", "intermediate", "intermediate"],
    "topics_seen": ["gradient descent", "neural networks", "backpropagation",
                    "transformers", "overfitting", "cnn"],
    "topic_counts": {
        "gradient descent": 5,
        "backpropagation": 4,
        "transformers": 3,
        "overfitting": 2,
        "neural networks": 4,
        "cnn": 2,
    },
    "weak_areas": {
        "backpropagation": ["chain rule", "gradient flow"],
        "transformers": ["query-key-value mechanics"],
        "gradient descent": ["learning rate scheduling"],
    },
    "mastery": {
        "gradient descent": 0.72,
        "neural networks": 0.65,
        "backpropagation": 0.41,
        "transformers": 0.55,
        "overfitting": 0.78,
        "cnn": 0.60,
    },
    "used_explanations": {
        "gradient descent": ["beginner-default", "intermediate-default"],
        "backpropagation": ["beginner-default", "intermediate-remedial"],
        "transformers": ["intermediate-default"],
        "neural networks": ["beginner-default", "intermediate-advance"],
    },
    "last_evaluation": {
        "topic": "backpropagation",
        "understanding_level": "partial",
        "weak_concepts": ["chain rule"],
        "recommended_action": "re-explain",
    },
})

BASELINE_PROFILE = ensure_profile_structure({})


def _truncate(text: str, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "…"


def run_question(question: str, profile: dict) -> dict:
    level, confidence, topic, examples, weak_examples, answer, teaching_mode = generate_tutor_response(
        question, profile
    )
    return {
        "level": level,
        "confidence": round(max(confidence) if isinstance(confidence, list) else confidence, 2),
        "topic": topic,
        "teaching_mode": teaching_mode,
        "answer_excerpt": _truncate(answer, max_words=55),
    }


def build_markdown_report(results: list) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# EduAgent Personalization Evaluation",
        "",
        f"Generated: {now}  ",
        "Baseline = empty profile &nbsp;|&nbsp; Personalized = simulated 8-session learner profile",
        "",
        "---",
        "",
    ]

    for i, entry in enumerate(results, 1):
        q = entry["question"]
        b = entry["baseline"]
        p = entry["personalized"]

        lines += [
            f"## Q{i}: {q}",
            "",
            "| Dimension | Baseline | Personalized |",
            "|-----------|----------|--------------|",
            f"| Detected Level | `{b['level']}` | `{p['level']}` |",
            f"| Confidence | {b['confidence']} | {p['confidence']} |",
            f"| Topic | {b['topic']} | {p['topic']} |",
            f"| Teaching Mode | **{b['teaching_mode']}** | **{p['teaching_mode']}** |",
            f"| Answer Excerpt | {b['answer_excerpt']} | {p['answer_excerpt']} |",
            "",
        ]

        # Highlight when personalization made a visible difference
        if b["teaching_mode"] != p["teaching_mode"]:
            lines.append(
                f"> **Personalization signal:** teaching mode changed "
                f"`{b['teaching_mode']}` → `{p['teaching_mode']}` based on learner history."
            )
            lines.append("")
        if b["level"] != p["level"]:
            lines.append(
                f"> **Level shift:** baseline detected `{b['level']}`, "
                f"personalized path detected `{p['level']}`."
            )
            lines.append("")

        lines.append("---")
        lines.append("")

    # Summary table
    mode_changes = sum(
        1 for e in results if e["baseline"]["teaching_mode"] != e["personalized"]["teaching_mode"]
    )
    level_changes = sum(
        1 for e in results if e["baseline"]["level"] != e["personalized"]["level"]
    )

    lines += [
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Questions evaluated | {len(results)} |",
        f"| Teaching mode changed by personalization | {mode_changes} / {len(results)} |",
        f"| Level detection differed | {level_changes} / {len(results)} |",
        "",
        "Teaching mode choices are driven by the learner's mastery score, weak areas, "
        "and previously used explanation styles stored in the profile. "
        "A higher fraction of mode changes indicates the memory system is actively "
        "adapting content to the individual learner.",
    ]

    return "\n".join(lines)


def main():
    print(f"Running evaluation on {len(TEST_QUESTIONS)} questions…\n")
    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question}")

        print("    → baseline…", end=" ", flush=True)
        baseline = run_question(question, dict(BASELINE_PROFILE))
        print(f"mode={baseline['teaching_mode']}")

        print("    → personalized…", end=" ", flush=True)
        personalized = run_question(question, dict(PERSONALIZED_PROFILE))
        print(f"mode={personalized['teaching_mode']}")

        results.append({
            "question": question,
            "baseline": baseline,
            "personalized": personalized,
        })

    report = build_markdown_report(results)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
