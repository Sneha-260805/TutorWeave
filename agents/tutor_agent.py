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
        # Truncate answers to keep the prompt tight and avoid timeout
        answer_short = str(row.answer)[:300].rstrip()
        parts.append(
            f"[{i}] Q: {row.question}\n"
            f"    A: {answer_short}\n"
            f"    Topic: {row.topic}"
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
    Draws only from retrieved examples for the actual topic — never injects canned content.
    """
    if examples_df is not None and len(examples_df) > 0:
        first = examples_df.iloc[0]
        example_answer = str(first.get("answer", "")).strip()
        if example_answer:
            return (
                f"The live tutor model is temporarily unavailable. "
                f"Here is a relevant explanation about **{topic}** from the knowledge base:\n\n"
                f"{example_answer}"
            )

    return (
        f"The live tutor model is temporarily unavailable. "
        f"Your question is about **{topic}** at **{level}** level. "
        f"Please try again in a moment for a full adaptive explanation."
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

    eval_hint = evaluation_strategy_hint or ""
    # Only inject memory hint when it carries real learner history (not the blank-profile noise)
    memory_is_informative = memory_hint and "no strong prior history" not in memory_hint.lower()
    memory_section = f"\nLearner context: {memory_hint}" if memory_is_informative else ""
    eval_section = f"\nStrategy hint: {eval_hint}" if eval_hint else ""

    system_message = (
        f"You are EduAgent, an adaptive AI tutor specializing in {topic}. "
        f"You MUST stay focused on {topic} throughout your answer. "
        f"Do NOT drift into unrelated ML concepts such as gradient descent, learning rate, "
        f"or optimization unless {topic} itself explicitly requires them. "
        f"Pitch your explanation at {level} level."
    )

    user_message = f"""QUESTION: {user_question}
TOPIC: {topic}{memory_section}{eval_section}

RETRIEVED KNOWLEDGE BASE EXAMPLES — these are your PRIMARY source:
{examples_text}{weak_rag_section}

GROUNDING RULES (follow strictly):
1. Build your answer directly upon the facts, concepts, and explanations in the retrieved examples above.
2. Every key claim in your answer should trace back to the retrieved examples.
3. You may expand or re-explain retrieved content — but do NOT ignore it.
4. Do NOT copy retrieved text verbatim — synthesize and adapt in your own words.
5. Do NOT introduce facts or concepts that contradict the retrieved examples.

LEVEL GUIDE:
- beginner: simple words, analogies, no jargon
- intermediate: moderate detail, 1-2 key terms explained
- advanced: technical depth, assume background knowledge

{mode_instruction}
Answer clearly and concisely (3–6 sentences)."""

    fallback_answer = build_local_fallback_answer(user_question, level, topic, examples)

    answer = complete_chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        fallback=fallback_answer,
        temperature=0.25,
        max_tokens=512,
    )
    return level, confidence, topic, examples, weak_examples, answer, teaching_mode
