from ml.classifier import predict_level
from ml.topic_detector import detect_best_topic
from ml.retriever import retrieve_examples, retrieve_for_weak_areas, df as RETRIEVER_DF
from agents.memory_agent import build_memory_hint, build_evaluation_strategy_hint
from agents.llm_client import complete_chat
from config.settings import RAG_TOP_N, RAG_WEAK_TOP_N


def format_examples(examples_df):
    if examples_df is None or len(examples_df) == 0:
        return "No examples found."

    parts = []
    for i, row in enumerate(examples_df.itertuples(index=False), 1):
        parts.append(
            f"Example {i}:\n"
            f"Question: {row.question}\n"
            f"Answer: {row.answer}\n"
            f"Topic: {row.topic}"
        )
    return "\n\n".join(parts)


def infer_teaching_mode(evaluation_strategy_hint: str, profile: dict = None, topic: str = "") -> str:
    """
    Select teaching mode.

    Priority order:
    1. last_evaluation hint (only present when the last eval was on the same topic)
    2. Mastery score + weak_areas from the learner profile
    3. Default fallback
    """
    hint_lower = evaluation_strategy_hint.lower()
    if "teaching mode: remedial" in hint_lower:
        return "remedial"
    if "teaching mode: clarification" in hint_lower:
        return "clarification"
    if "teaching mode: advance" in hint_lower:
        return "advance"

    # Fall back to mastery-based selection when last_eval hint is absent
    if profile and topic:
        from agents.memory_agent import MASTERY_INITIAL_SCORE, _safe_mastery_score
        mastery = _safe_mastery_score(
            profile.get("mastery", {}).get(topic), default=MASTERY_INITIAL_SCORE
        )
        weak_areas = profile.get("weak_areas", {}).get(topic, [])
        topic_count = profile.get("topic_counts", {}).get(topic, 0)

        if mastery < 0.45 and (weak_areas or topic_count >= 3):
            return "remedial"
        if weak_areas and mastery <= 0.75:
            return "clarification"
        if mastery > 0.75 and topic_count >= 2:
            return "advance"

    return "default"


def build_mode_specific_instruction(teaching_mode: str) -> str:
    """
    Return strong tutor instructions based on teaching mode.
    """
    if teaching_mode == "remedial":
        return """
Mode-specific instructions:
- Re-teach from scratch.
- Use very simple wording.
- Use one small concrete example.
- Explain in 4 to 6 short sentences.
- Focus on one main idea first before adding detail.
- Avoid abstract definitions unless absolutely necessary.
- Avoid repeating the same explanation style or analogy used earlier for this topic.
- Start with: "Let's simplify it completely."
"""
    elif teaching_mode == "clarification":
        return """
Mode-specific instructions:
- Briefly restate the main idea in one or two simple sentences.
- Then focus on the weak or confusing concept.
- Use one clear example.
- Do not repeat the whole long explanation.
- Keep the answer focused and moderately short.
"""
    elif teaching_mode == "advance":
        return """
Mode-specific instructions:
- Assume the learner understood the basics.
- Avoid repeating the full beginner introduction.
- Give a slightly deeper explanation.
- Connect the concept to a related next-step idea.
- Keep the answer clear but more intellectually rich.
"""
    else:
        return """
Mode-specific instructions:
- Give a normal level-appropriate explanation.
"""


def build_local_fallback_answer(user_question: str, level: str, topic: str, examples_df) -> str:
    """
    Return a quick non-LLM answer when the external tutor model is unavailable.
    """
    example_hint = ""
    if examples_df is not None and len(examples_df) > 0:
        first = examples_df.iloc[0]
        example_hint = f"\n\nA related dataset example says: {first.get('answer', '')}"

    return (
        "The live tutor model is taking too long, so here is a quick local explanation.\n\n"
        f"Your question is about **{topic}**, and it was detected as **{level}** level. "
        f"For your question, \"{user_question}\", focus on the core relationship between the main concept "
        "and the outcome you are asking about. If the question is about gradient descent, remember that "
        "learning rate controls step size: a very small rate learns slowly, a reasonable rate converges "
        "steadily, and a very large rate can overshoot or become unstable."
        f"{example_hint}"
    )


def generate_tutor_response(user_question: str, profile: dict):
    level, confidence = predict_level(user_question)
    topic = detect_best_topic(user_question, level, RETRIEVER_DF)
    if not topic:
        topic = "general"

    # Primary RAG retrieval — semantically closest to the user's question
    examples = retrieve_examples(user_question, level, top_n=RAG_TOP_N)
    examples_text = format_examples(examples)

    # Personalized RAG retrieval — examples targeting the learner's weak concepts
    weak_concepts = profile.get("weak_areas", {}).get(topic, [])
    already_retrieved = set(examples["question"].tolist()) if len(examples) > 0 else set()
    weak_examples = retrieve_for_weak_areas(
        weak_concepts, topic, level, top_n=RAG_WEAK_TOP_N, exclude_questions=already_retrieved
    )
    weak_examples_text = format_examples(weak_examples) if len(weak_examples) > 0 else ""

    memory_hint = build_memory_hint(profile, topic)
    evaluation_strategy_hint = build_evaluation_strategy_hint(profile, topic)
    teaching_mode = infer_teaching_mode(evaluation_strategy_hint, profile, topic)
    mode_instruction = build_mode_specific_instruction(teaching_mode)

    weak_rag_section = (
        f"\nRetrieved examples targeting your weak areas ({', '.join(weak_concepts)}):\n{weak_examples_text}"
        if weak_examples_text
        else ""
    )

    prompt = f"""
You are EduAgent, an adaptive AI tutor.

Student question:
{user_question}

Detected student level:
{level}

Detected topic:
{topic}

Learner memory hint:
{memory_hint}

Recent evaluator strategy hint:
{evaluation_strategy_hint if evaluation_strategy_hint else "No recent evaluation strategy available for this topic."}

Retrieved dataset examples (semantically similar to the student's question):
{examples_text}{weak_rag_section}

General instructions:
- Answer according to the detected level.
- For beginner: use simple words, intuition, and easy examples.
- For intermediate: explain clearly with moderate detail and 1-2 key technical terms.
- For advanced: give a deeper, more technical explanation.
- Treat retrieved examples as supporting context, not as guaranteed ground truth.
- Use relevant examples to ground the explanation style and topic coverage.
- Do not copy the retrieved examples directly.
- Use learner memory to avoid repeating the same explanation style.
- If weak area examples are provided, use them to specifically address those gaps.
- If recent evaluator strategy exists, adapt the answer accordingly.
- Keep the answer educational, structured, and concise.

{mode_instruction}

Now answer the student's question.
"""

    fallback_answer = build_local_fallback_answer(user_question, level, topic, examples)

    answer = complete_chat(
        [{"role": "user", "content": prompt}],
        fallback=fallback_answer,
        temperature=0.25,
    )
    return level, confidence, topic, examples, weak_examples, answer, teaching_mode
