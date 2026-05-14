import json
import logging
from html import escape

import gradio as gr
import matplotlib.pyplot as plt

from auth.auth_service import signup_user, login_user
from db.profile_repository import load_profile, save_profile
from db.sqlite_store import init_db
from agents.tutor_agent import generate_tutor_response
from agents.evaluator_agent import (
    generate_followup_question,
    evaluate_followup_response,
)
from agents.memory_agent import (
    ensure_profile_structure,
    update_profile_after_question,
    update_profile_after_evaluation,
    update_last_evaluation,
    record_used_explanation,
    build_memory_hint,
    build_evaluation_strategy_hint,
)
from app.ui import build_demo
from voice.utils import transcribe_audio_path, synthesize_text_to_wav


def confidence_to_text(conf):
    return (
        f"Beginner: {conf[0]:.3f} | "
        f"Intermediate: {conf[1]:.3f} | "
        f"Advanced: {conf[2]:.3f}"
    )


def _format_example_rows(examples_df, offset: int = 0) -> list:
    lines = []
    for i, row in enumerate(examples_df.itertuples(index=False), offset + 1):
        lines.append(
            f"### Example {i}\n"
            f"**Question:** {row.question}\n\n"
            f"**Answer:** {row.answer}\n\n"
            f"**Level:** {row.level}\n\n"
            f"**Topic:** {row.topic}"
        )
    return lines


def examples_to_markdown(examples_df, weak_examples_df=None):
    if examples_df is None or len(examples_df) == 0:
        return "No examples retrieved."

    lines = ["## Retrieved Examples (Semantically Similar to Your Question)"]
    lines += _format_example_rows(examples_df)

    if weak_examples_df is not None and len(weak_examples_df) > 0:
        lines.append("\n## Retrieved Examples (Targeting Your Weak Areas)")
        lines += _format_example_rows(weak_examples_df, offset=len(examples_df))
    return "\n---\n".join(lines)


def profile_to_markdown(profile):
    profile = ensure_profile_structure(profile)

    def _list_items(values):
        values = [escape(str(value)) for value in values if value]
        if not values:
            return "<p>None yet</p>"
        return "<ul>" + "".join(f"<li>{value}</li>" for value in values) + "</ul>"

    def _topic_map_items(topic_map):
        rows = []
        for topic, values in topic_map.items():
            if values:
                concepts = ", ".join(escape(str(value)) for value in values)
                rows.append(f"<li><strong>{escape(str(topic))}</strong>: {concepts}</li>")
        if not rows:
            return "<p>None yet</p>"
        return "<ul>" + "".join(rows) + "</ul>"

    weak_areas_lines = []
    for topic, concepts in profile.get("weak_areas", {}).items():
        if concepts:
            weak_areas_lines.append((topic, concepts))

    mastery_lines = []
    for topic, score in profile.get("mastery", {}).items():
        try:
            score_text = f"{float(score):.2f}"
        except (TypeError, ValueError):
            score_text = escape(str(score))
        mastery_lines.append(f"{escape(str(topic))}: {score_text}")

    recommendations = profile.get("recommended_next_topics", [])
    topics_seen = profile.get("topics_seen", [])

    weak_areas = _topic_map_items(dict(weak_areas_lines))

    return (
        "<div class='profile-grid'>"
        "<div class='profile-card'>"
        "<h4>Activity</h4>"
        f"<p><strong>Sessions:</strong> {profile.get('sessions', 0)}<br>"
        f"<strong>Questions Asked:</strong> {profile.get('questions_asked', 0)}<br>"
        f"<strong>Last Level:</strong> {escape(str(profile.get('last_level', 'beginner')))}</p>"
        "</div>"
        "<div class='profile-card'>"
        "<h4>Topics Seen</h4>"
        f"{_list_items(topics_seen)}"
        "</div>"
        "<div class='profile-card wide-card'>"
        "<h4>Weak Areas</h4>"
        f"{weak_areas}"
        "</div>"
        "<div class='profile-card'>"
        "<h4>Mastery</h4>"
        f"{_list_items(mastery_lines)}"
        "</div>"
        "<div class='profile-card'>"
        "<h4>Recommended Next Topics</h4>"
        f"{_list_items(recommendations)}"
        "</div>"
        "</div>"
    )


def format_evaluation_markdown(evaluation: dict):
    level = (evaluation.get("understanding_level", "partial") or "partial").lower()
    if level == "good":
        css_class = "eval-good"
        label = "GOOD"
    elif level == "poor":
        css_class = "eval-poor"
        label = "POOR"
    else:
        css_class = "eval-partial"
        label = "PARTIAL"

    weak_concepts = evaluation.get("weak_concepts", [])
    weak_text = ", ".join(escape(str(item)) for item in weak_concepts) if weak_concepts else "None"

    return (
        f"<div class='eval-card {css_class}'>"
        f"<h3>Understanding Level: {label}</h3>"
        f"<p><strong>Weak Concepts:</strong> {weak_text}</p>"
        f"<p><strong>Feedback:</strong> {escape(str(evaluation.get('feedback', '')))}</p>"
        f"<p><strong>Recommended Action:</strong> {escape(str(evaluation.get('recommended_action', '')))}</p>"
        "</div>"
    )


CHART_BG = "#0b1120"
CHART_PANEL = "#111827"
CHART_TEXT = "#e5e7eb"
CHART_MUTED = "#94a3b8"
CHART_GRID = "#334155"


def _style_dark_axis(ax, title: str):
    ax.set_facecolor(CHART_PANEL)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    ax.tick_params(axis="x", colors=CHART_MUTED, labelsize=9, rotation=25)
    ax.tick_params(axis="y", colors=CHART_MUTED, labelsize=9)
    ax.yaxis.label.set_color(CHART_MUTED)
    ax.grid(axis="y", color=CHART_GRID, alpha=0.34, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#243044")


def _placeholder_chart(title: str, message: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_PANEL)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color=CHART_MUTED)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#f8fafc", pad=12)
    fig.tight_layout(pad=1.4)
    plt.close(fig)
    return fig


def build_mastery_chart(profile: dict):
    profile = ensure_profile_structure(profile)
    mastery = profile.get("mastery", {})
    if not mastery:
        return _placeholder_chart("Mastery by Topic", "No mastery data yet.\nAsk and evaluate a few questions.")

    chart_rows = []
    for topic, score in mastery.items():
        try:
            chart_rows.append((topic, float(score)))
        except (TypeError, ValueError):
            continue
    if not chart_rows:
        return _placeholder_chart("Mastery by Topic", "Mastery data is not numeric yet.")

    topics = [topic for topic, _score in chart_rows]
    scores = [score for _topic, score in chart_rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    bars = ax.bar(topics, scores, color="#14b8a6", edgecolor="#99f6e4", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=3, color=CHART_TEXT, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mastery Score")
    _style_dark_axis(ax, "Mastery by Topic")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.4)
    plt.close(fig)
    return fig


def build_topic_revisit_chart(profile: dict):
    profile = ensure_profile_structure(profile)
    topic_counts = profile.get("topic_counts", {})
    if not topic_counts:
        return _placeholder_chart("Topic Revisit Count", "No topic revisit data yet.")

    chart_rows = []
    for topic, count in topic_counts.items():
        try:
            chart_rows.append((topic, int(count)))
        except (TypeError, ValueError):
            continue
    if not chart_rows:
        return _placeholder_chart("Topic Revisit Count", "Topic revisit data is not numeric yet.")

    topics = [topic for topic, _count in chart_rows]
    counts = [count for _topic, count in chart_rows]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    bars = ax.bar(topics, counts, color="#3b82f6", edgecolor="#93c5fd", linewidth=0.8)
    ax.bar_label(bars, padding=3, color=CHART_TEXT, fontsize=8)
    if counts:
        ax.set_ylim(0, max(counts) * 1.22 + 0.2)
    ax.set_ylabel("Questions/Revisits")
    _style_dark_axis(ax, "Topic Revisit Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.4)
    plt.close(fig)
    return fig


def build_weak_concepts_chart(profile: dict):
    profile = ensure_profile_structure(profile)
    weak_areas = profile.get("weak_areas", {})
    counts = {topic: len(concepts or []) for topic, concepts in weak_areas.items() if concepts}
    if not counts:
        return _placeholder_chart("Weak Concept Count by Topic", "No weak-concept signals yet.")

    topics = list(counts.keys())
    values = [counts[t] for t in topics]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    fig.patch.set_facecolor(CHART_BG)
    bars = ax.bar(topics, values, color="#f59e0b", edgecolor="#fde68a", linewidth=0.8)
    ax.bar_label(bars, padding=3, color=CHART_TEXT, fontsize=8)
    if values:
        ax.set_ylim(0, max(values) * 1.22 + 0.2)
    ax.set_ylabel("Weak Concept Count")
    _style_dark_axis(ax, "Weak Concept Count by Topic")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=1.4)
    plt.close(fig)
    return fig


def build_system_insights_markdown(
    level: str = "",
    confidence_text: str = "",
    topic: str = "",
    profile: dict | None = None,
):
    profile = ensure_profile_structure(profile or {})
    memory_hint = build_memory_hint(profile, topic) if topic else "No topic yet."
    eval_hint = build_evaluation_strategy_hint(profile, topic) if topic else "No topic yet."
    last_eval = profile.get("last_evaluation", {})
    last_eval_summary = "No evaluation yet."
    if last_eval:
        last_eval_summary = (
            f"{last_eval.get('understanding_level', 'partial')} on "
            f"{last_eval.get('topic', 'unknown topic')}: "
            f"{last_eval.get('recommended_action', 'give more practice')}"
        )

    return (
        "<div class='insight-block'>"
        "<h4>Pipeline Signals</h4>"
        f"<p><strong>Predicted Level:</strong> {escape(str(level or 'N/A'))}<br>"
        f"<strong>Confidence Scores:</strong> {escape(str(confidence_text or 'N/A'))}<br>"
        f"<strong>Detected Topic:</strong> {escape(str(topic or 'N/A'))}</p>"
        "</div>"
        "<div class='insight-block'>"
        "<h4>Memory Hint / Tutor Strategy</h4>"
        f"<p>{escape(memory_hint)}</p>"
        "</div>"
        "<div class='insight-block'>"
        "<h4>Last Evaluation Summary</h4>"
        f"<p>{escape(last_eval_summary)}</p>"
        "</div>"
        "<div class='insight-block'>"
        "<h4>Evaluator Strategy Hint</h4>"
        f"<p>{escape(str(eval_hint)) if eval_hint else 'No strategy hint yet.'}</p>"
        "</div>"
        "<div class='insight-block'>"
        "<h4>Last Evaluation JSON</h4>"
        f"<pre><code>{escape(json.dumps(last_eval, indent=2))}</code></pre>"
        "</div>"
    )


# ---------------------------------------------------------
# QUESTION FLOW
# ---------------------------------------------------------
def handle_question(user_state, chat_history, user_question):
    """
    Handle main tutor question.

    user_state: current logged-in user dict
    chat_history: current chatbot history
    user_question: learner input

    Returns:
    chatbot_history,
    followup_context,
    detected_level,
    confidence_text,
    detected_topic,
    tutor_answer,
    followup_question,
    examples_markdown,
    profile_markdown,
    evaluation_markdown,
    status_message
    """
    if chat_history is None:
        chat_history = []

    if not user_state:
        return (
            chat_history,
            None,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Please login first."
        )

    if not user_question or not user_question.strip():
        return (
            chat_history,
            None,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "Please enter a question."
        )

    user_id = user_state["id"]
    profile = load_profile(user_id)
    profile = ensure_profile_structure(profile)

    # Tutor Agent
    level, confidence, topic, examples, weak_examples, answer, teaching_mode = generate_tutor_response(user_question, profile)

    # Memory update after question
    profile = update_profile_after_question(profile, topic, level)
    profile = record_used_explanation(profile, topic, f"{level}-{teaching_mode}")
    save_profile(user_id, profile)

    # Evaluator Agent: generate follow-up
    followup_question = generate_followup_question(
        user_question=user_question,
        tutor_answer=answer,
        level=level,
        topic=topic
    )

    bot_reply = (
        f"**Answer:**\n{answer}\n\n"
        f"**Check your understanding:**\n{followup_question}"
    )

    chat_history = chat_history + [
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": bot_reply},
    ]

    # Store context needed when learner replies to follow-up
    followup_context = {
        "topic": topic,
        "level": level,
        "followup_question": followup_question,
        "last_user_question": user_question,
        "last_tutor_answer": answer,
    }

    return (
        chat_history,
        followup_context,
        level,
        confidence_to_text(confidence),
        topic,
        answer,
        followup_question,
        examples_to_markdown(examples, weak_examples),
        profile_to_markdown(profile),
        "",
        "Question answered successfully."
    )


# ---------------------------------------------------------
# FOLLOW-UP EVALUATION FLOW
# ---------------------------------------------------------
def handle_followup_reply(user_state, followup_context, learner_reply):
    if not user_state:
        return "", "", "Please login first."

    if not followup_context:
        return "", "", "No follow-up question found yet. Ask a main question first."

    if not learner_reply or not learner_reply.strip():
        return "", "", "Please enter your reply to the follow-up question."

    user_id = user_state["id"]
    profile = load_profile(user_id)
    profile = ensure_profile_structure(profile)

    topic = followup_context["topic"]
    level = followup_context["level"]
    followup_question = followup_context["followup_question"]

    evaluation = evaluate_followup_response(
        topic=topic,
        level=level,
        followup_question=followup_question,
        learner_reply=learner_reply
    )

    profile = update_profile_after_evaluation(profile, topic, evaluation)
    profile = update_last_evaluation(profile, topic, evaluation)
    save_profile(user_id, profile)

    return (
        format_evaluation_markdown(evaluation),
        profile_to_markdown(profile),
        "Follow-up evaluated successfully."
    )
# ---------------------------------------------------------
# SESSION / RESET HELPERS
# ---------------------------------------------------------
def clear_chat_and_followup(user_state):
    """
    Clear only the visible chat and follow-up context.
    Keep user profile persisted.
    """
    profile_md = ""
    if user_state:
        profile = load_profile(user_state["id"])
        profile = ensure_profile_structure(profile)
        profile_md = profile_to_markdown(profile)

    return (
        [],        # chat_history
        None,      # followup_context
        "", "", "", "", "", "",  # detected level / conf / topic / answer / followup / examples
        profile_md,
        "",        # evaluation markdown
        "Chat cleared."
    )


# ---------------------------------------------------------
# APP CREATION
# ---------------------------------------------------------

# Output key lists — must match the outputs= lists in ui.py exactly.
# _pack() validates length and order at runtime so mismatches are caught immediately.
_SIGNUP_KEYS = [
    "auth_status", "signup_name", "signup_username", "signup_email",
    "signup_password", "auth_tabs", "login_identifier",
]
_AUTH_KEYS = [
    "auth_status", "logged_in", "user_state", "welcome_md", "chatbot", "state",
    "level_box", "conf_box", "topic_box", "profile_md", "insights_md",
    "mastery_plot", "revisit_plot", "weak_concept_plot",
    "auth_section", "app_section", "followup_context_state",
]
_ASK_KEYS = [
    "chatbot", "state", "followup_context_state",
    "level_box", "conf_box", "topic_box", "followup_box",
    "profile_md", "examples_md", "insights_md",
    "mastery_plot", "revisit_plot", "weak_concept_plot",
]
_EVAL_KEYS = [
    "evaluation_md", "profile_md", "eval_status",
    "insights_md", "mastery_plot", "revisit_plot", "weak_concept_plot",
]


def _pack(outputs: dict, keys: list) -> tuple:
    missing = [k for k in keys if k not in outputs]
    extra = [k for k in outputs if k not in keys]
    if missing or extra:
        raise ValueError(f"Handler output mismatch — missing: {missing}, unexpected: {extra}")
    return tuple(outputs[k] for k in keys)


def create_app():
    """
    Create the Gradio application.
    """
    init_db()

    def handle_signup(name, username, email, password):
        ok, message = signup_user(name=name, email=email, password=password, username=username)
        if ok:
            return _pack({
                "auth_status": message,
                "signup_name": "",
                "signup_username": "",
                "signup_email": "",
                "signup_password": "",
                "auth_tabs": gr.update(selected=0),
                "login_identifier": email,
            }, _SIGNUP_KEYS)
        return _pack({
            "auth_status": message,
            "signup_name": name,
            "signup_username": username or "",
            "signup_email": email,
            "signup_password": "",
            "auth_tabs": gr.update(),
            "login_identifier": gr.update(),
        }, _SIGNUP_KEYS)

    def handle_login(identifier, password):
        success, message, user = login_user(email=identifier, password=password)
        if not success:
            return _pack({
                "auth_status": message,
                "logged_in": False,
                "user_state": None,
                "welcome_md": "",
                "chatbot": [],
                "state": [],
                "level_box": "",
                "conf_box": "",
                "topic_box": "",
                "profile_md": "",
                "insights_md": "",
                "mastery_plot": _placeholder_chart("Mastery by Topic", "Login to view mastery trends."),
                "revisit_plot": _placeholder_chart("Topic Revisit Count", "Login to view topic revisits."),
                "weak_concept_plot": _placeholder_chart("Weak Concept Count by Topic", "Login to view weak concepts."),
                "auth_section": gr.update(visible=True),
                "app_section": gr.update(visible=False),
                "followup_context_state": None,
            }, _AUTH_KEYS)
        profile = load_profile(user["id"])
        profile = ensure_profile_structure(profile)
        profile["sessions"] += 1
        save_profile(user["id"], profile)
        return _pack({
            "auth_status": message,
            "logged_in": True,
            "user_state": user,
            "welcome_md": f"Logged in as **{user['name']}** ({user['email']})",
            "chatbot": [],
            "state": [],
            "level_box": "",
            "conf_box": "",
            "topic_box": "",
            "profile_md": profile_to_markdown(profile),
            "insights_md": build_system_insights_markdown(profile=profile),
            "mastery_plot": build_mastery_chart(profile),
            "revisit_plot": build_topic_revisit_chart(profile),
            "weak_concept_plot": build_weak_concepts_chart(profile),
            "auth_section": gr.update(visible=False),
            "app_section": gr.update(visible=True),
            "followup_context_state": None,
        }, _AUTH_KEYS)

    def handle_logout():
        return _pack({
            "auth_status": "Logged out.",
            "logged_in": False,
            "user_state": None,
            "welcome_md": "",
            "chatbot": [],
            "state": [],
            "level_box": "",
            "conf_box": "",
            "topic_box": "",
            "profile_md": "",
            "insights_md": "",
            "mastery_plot": _placeholder_chart("Mastery by Topic", "Login to view mastery trends."),
            "revisit_plot": _placeholder_chart("Topic Revisit Count", "Login to view topic revisits."),
            "weak_concept_plot": _placeholder_chart("Weak Concept Count by Topic", "Login to view weak concepts."),
            "auth_section": gr.update(visible=True),
            "app_section": gr.update(visible=False),
            "followup_context_state": None,
        }, _AUTH_KEYS)

    def ask_eduagent(user_question, chat_history, user):
        (
            chatbot_history, followup_context, level, confidence_text, topic,
            _tutor_answer, followup_question, examples_markdown,
            profile_markdown, _evaluation_markdown, status_message,
        ) = handle_question(user, chat_history, user_question)

        profile_obj = ensure_profile_structure(
            load_profile(user["id"]) if user else {}
        )
        return _pack({
            "chatbot": chatbot_history,
            "state": chatbot_history,
            "followup_context_state": followup_context,
            "level_box": level,
            "conf_box": confidence_text,
            "topic_box": topic,
            "followup_box": followup_question,
            "profile_md": profile_markdown,
            "examples_md": examples_markdown,
            "insights_md": build_system_insights_markdown(level, confidence_text, topic, profile_obj),
            "mastery_plot": build_mastery_chart(profile_obj),
            "revisit_plot": build_topic_revisit_chart(profile_obj),
            "weak_concept_plot": build_weak_concepts_chart(profile_obj),
        }, _ASK_KEYS)

    def clear_chat(user_state=None):
        if user_state:
            profile_obj = ensure_profile_structure(load_profile(user_state["id"]))
            profile_markdown = profile_to_markdown(profile_obj)
        else:
            profile_obj = ensure_profile_structure({})
            profile_markdown = ""
        return _pack({
            "chatbot": [],
            "state": [],
            "followup_context_state": None,
            "level_box": "",
            "conf_box": "",
            "topic_box": "",
            "followup_box": "",
            "profile_md": profile_markdown,
            "examples_md": "",
            "insights_md": build_system_insights_markdown(profile=profile_obj),
            "mastery_plot": build_mastery_chart(profile_obj),
            "revisit_plot": build_topic_revisit_chart(profile_obj),
            "weak_concept_plot": build_weak_concepts_chart(profile_obj),
        }, _ASK_KEYS)

    def handle_eval_reply(user_state, followup_context, learner_reply):
        evaluation_md, updated_profile_md, status_message = handle_followup_reply(
            user_state, followup_context, learner_reply
        )
        profile_obj = (
            ensure_profile_structure(load_profile(user_state["id"]))
            if user_state else ensure_profile_structure({})
        )
        topic = followup_context.get("topic", "") if followup_context else ""
        level = followup_context.get("level", "") if followup_context else ""
        return _pack({
            "evaluation_md": evaluation_md,
            "profile_md": updated_profile_md,
            "eval_status": status_message,
            "insights_md": build_system_insights_markdown(level=level, topic=topic, profile=profile_obj),
            "mastery_plot": build_mastery_chart(profile_obj),
            "revisit_plot": build_topic_revisit_chart(profile_obj),
            "weak_concept_plot": build_weak_concepts_chart(profile_obj),
        }, _EVAL_KEYS)

    def transcribe_question_handler(audio_path, current_text):
        """Transcribe a recorded question into the normal Ask textbox."""
        if not audio_path:
            return current_text or ""

        transcription, _confidence = transcribe_audio_path(audio_path)
        if not transcription:
            gr.Warning("Could not transcribe the recording. Please try again or type your question.")
            return current_text or ""
        return transcription

    def read_answer_handler(last_answer_text):
        """Read the answer text aloud"""
        if not last_answer_text:
            return ""
        try:
            audio_path = synthesize_text_to_wav(last_answer_text)
            return audio_path or ""
        except Exception as e:
            logging.error(f"TTS failed: {e}")
            return ""

    return build_demo(
        handle_signup=handle_signup,
        handle_login=handle_login,
        handle_logout=handle_logout,
        ask_eduagent=ask_eduagent,
        clear_chat=clear_chat,
        handle_followup_reply=handle_eval_reply,
        transcribe_question_handler=transcribe_question_handler,
        read_answer_handler=read_answer_handler,
    )
