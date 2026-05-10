# EduAgent — Architecture Reference

## Overview

EduAgent is an adaptive AI tutoring system for machine learning and AI concepts. It combines a fine-tuned DistilBERT classifier, semantic retrieval, a Groq-hosted LLM, and a persistent learner profile to deliver explanations that evolve with the individual learner.

The system is built around three cooperating agents, each with a single clear responsibility, coordinated by a thin Gradio application layer.

---

## Agent Flow Diagram

```
User types a question
        │
        ▼
┌───────────────────┐
│   ml/classifier   │  DistilBERT (fine-tuned)
│  predict_level()  │  → beginner / intermediate / advanced
│                   │    + confidence score
└────────┬──────────┘
         │  level, confidence
         ▼
┌───────────────────┐
│ ml/topic_detector │  TF-IDF cosine over unique topics in dataset
│ detect_best_topic │  + alias expansion (llm→large language models, etc.)
└────────┬──────────┘
         │  topic
         ▼
┌───────────────────┐
│  ml/retriever     │  sentence-transformers all-MiniLM-L6-v2 (dense)
│retrieve_examples()│  or TF-IDF fallback if package absent
│                   │  → top-2 (question, answer) pairs from dataset
│  Index cache per  │    filtered by level + topic slice
│  (level, topic)   │
└────────┬──────────┘
         │  examples_df
         ▼
┌───────────────────┐
│ agents/memory_    │  Reads learner profile from DB
│    agent          │  build_memory_hint()          → text injected into prompt
│                   │  build_evaluation_strategy_   → teaching mode hint
│                   │    hint()
└────────┬──────────┘
         │  memory_hint, strategy_hint
         ▼
┌───────────────────┐
│ agents/tutor_     │  infer_teaching_mode()
│    agent          │    → remedial / clarification / advance / default
│                   │  Assembles LLM prompt:
│                   │    level + topic + examples
│                   │    + memory_hint + strategy_hint
│                   │    + mode-specific instructions
│                   │  complete_chat() → Groq llama-3.3-70b-versatile
└────────┬──────────┘
         │  answer, teaching_mode
         ▼
    Shown to learner
         │
         │  (after learner responds)
         ▼
┌───────────────────┐
│ agents/evaluator_ │  generate_followup_question()
│    agent          │  evaluate_user_response()
│                   │  → EvaluationResult(Pydantic):
│                   │      understanding_level: good/partial/poor
│                   │      weak_concepts: [str]
│                   │      feedback: str
│                   │      recommended_action: advance/re-explain/…
└────────┬──────────┘
         │  evaluation dict
         ▼
┌───────────────────┐
│ agents/memory_    │  update_profile_after_evaluation()
│    agent          │    mastery += GAIN_RATE * (1 − mastery)   [good]
│                   │    mastery += DELTA_PARTIAL / DELTA_POOR  [partial/poor]
│                   │  record_used_explanation(topic, level-mode tag)
│                   │  save_profile() → SQLite (WAL journal)
└───────────────────┘
```

---

## Component Breakdown

### 1. Difficulty Classifier — `ml/classifier.py`

| Detail | Value |
|--------|-------|
| Model | DistilBERT (`distilbert-base-uncased`) fine-tuned on EduAgent dataset |
| Labels | `beginner`, `intermediate`, `advanced` |
| Path | `models/distilbert_eduagent_v2/` |
| Fallback | Keyword heuristic (`_heuristic_predict`) if model missing |

**Design decision — why fine-tune vs. zero-shot?**
A zero-shot LLM call would add 1–2 s latency per question. The fine-tuned DistilBERT classifies in ~50 ms locally. Because difficulty classification is a stable, closed-label problem (only three classes), supervised fine-tuning outperforms prompting in both speed and consistency.

**Known limitation:** The heuristic fallback fires for very short beginner-intent phrases (≤8 words). Questions phrased as commands ("Explain backprop") are sometimes misclassified as intermediate.

---

### 2. Topic Detector — `ml/topic_detector.py`

TF-IDF cosine similarity over the unique topic strings present in the dataset, with an alias expansion table (e.g. `llm → large language models`, `cnn → convolutional neural networks`). Topic detection runs entirely in-process with no model load — it is fast enough that no caching is needed.

---

### 3. Semantic Retriever — `ml/retriever.py`

| Detail | Value |
|--------|-------|
| Primary backend | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim dense embeddings) |
| Fallback backend | TF-IDF cosine similarity (`sklearn`) |
| Index granularity | One index per `(level, topic)` pair, built once and cached in memory |
| Returned columns | `question`, `answer`, `level`, `topic` |

**Why semantic retrieval beats TF-IDF for this task:**
TF-IDF matches keywords literally — "neural net training" and "how backprop works" share almost no vocabulary, so TF-IDF returns random-looking results. `all-MiniLM-L6-v2` encodes semantic meaning, so both map to nearby vectors and the retriever returns genuinely related examples. This matters because the retrieved examples are injected into the tutor prompt as grounding context.

**Complexity penalty:** Each question vector is penalised by a small score deduction if the question text contains graduate-level terms (`theorem`, `subgradient`, etc.) or is longer than 18 words. This steers the retriever away from mismatched advanced examples when a beginner asks a broadly phrased question.

---

### 4. Memory Agent — `agents/memory_agent.py`

The memory agent owns the learner profile schema and all mutations to it.

**Profile schema (key fields):**

```python
{
  "sessions": int,                  # login count
  "questions_asked": int,
  "last_level": str,                # beginner / intermediate / advanced
  "topics_seen": [str],             # ordered first-seen list
  "topic_counts": {topic: int},     # how many questions per topic
  "mastery": {topic: float},        # 0.0 – 1.0, starts at 0.5
  "weak_areas": {topic: [str]},     # concept-level gaps identified by evaluator
  "used_explanations": {topic: [str]},  # "level-mode" tags, e.g. "intermediate-advance"
  "level_history": [str],           # per-question level sequence
  "recommended_next_topics": [str],
  "last_evaluation": dict,          # most recent EvaluationResult
}
```

**Mastery update — diminishing returns:**

```
good:    mastery ← mastery + 0.20 × (1 − mastery)   # gains shrink near 1.0
partial: mastery ← mastery − 0.06
poor:    mastery ← mastery − 0.18
```

Mastery is clamped to `[0.0, 1.0]`. Starting value for a new topic is 0.5.

**Teaching mode selection** (`infer_teaching_mode`):

| Condition | Mode |
|-----------|------|
| Evaluator recommended re-explain, mastery < 0.45 | `remedial` |
| Weak concepts identified but mastery ≥ 0.45 | `clarification` |
| Mastery > 0.75, evaluator recommended advance | `advance` |
| Otherwise | `default` |

The selected mode changes the mode-specific instructions injected into the tutor prompt, directly shaping explanation depth and style.

**`used_explanations` deduplication:** Before the tutor generates a response, the memory hint lists explanation styles already used for the topic (e.g. `"intermediate-advance"`). The tutor prompt instructs the LLM to avoid repeating the same approach, pushing it toward a fresh angle on repeated visits.

---

### 5. Tutor Agent — `agents/tutor_agent.py`

Assembles and fires the main LLM prompt. Returns `(level, confidence, topic, examples_df, answer, teaching_mode)`.

The prompt is structured in layers so each layer can be independently updated:

1. **Identity** — "You are EduAgent, an adaptive AI tutor."
2. **Classification context** — level, topic
3. **Learner memory** — memory hint + evaluation strategy hint (from memory agent)
4. **Retrieved examples** — top-2 semantically similar Q&A pairs
5. **General instructions** — level-appropriate tone guidelines
6. **Mode-specific instructions** — one of four tailored instruction blocks

**LLM:** Groq `llama-3.3-70b-versatile`, temperature 0.25 (low variance — we want reliable educational output, not creative variation). A `build_local_fallback_answer()` function returns a template-based answer if the API call times out, preventing hard failures.

---

### 6. Evaluator Agent — `agents/evaluator_agent.py`

After the learner answers the follow-up question, the evaluator agent calls the LLM and expects a JSON response conforming to:

```python
class EvaluationResult(BaseModel):
    understanding_level: Literal["good", "partial", "poor"]
    weak_concepts: list[str]
    feedback: str
    recommended_action: Literal["advance", "re-explain",
                                "give easier example", "give more practice"]
```

Pydantic validation ensures the LLM response is always structurally safe before it touches the profile. A regex-based JSON extractor handles cases where the LLM wraps the JSON in markdown fences.

---

### 7. Application Layer — `app/main.py` + `app/ui.py`

Built with Gradio 6.0. Key design points:

- **`_pack(outputs_dict, KEYS)`** validates every handler's output tuple against a named key list at runtime. A mismatch raises `ValueError` immediately rather than silently routing the wrong value to the wrong UI component.
- **Auth tabs** switch automatically from Signup → Login tab on successful registration, and pre-fill the login identifier field with the registered email.
- **Ask button loading state** — button text changes to "Thinking…" and becomes non-interactive during inference, then resets. Prevents double-submissions.
- **No dead code** — the previous `build_ui()` function (~120 lines) and module-level handler aliases were removed.

---

### 8. Auth & Persistence — `auth/` + `db/`

| Layer | Implementation |
|-------|----------------|
| Password hashing | PBKDF2-SHA256 via `passlib.hash.pbkdf2_sha256` |
| Email validation | Regex-free split check (`_is_valid_email`) |
| Database | SQLite with **WAL journal mode** |
| Profile storage | JSON blob in `profiles` table, loaded at login, saved after each evaluation |
| Race condition guard | `INSERT OR IGNORE` + rollback on `create_profile_if_missing` |

WAL (Write-Ahead Logging) was chosen over the default DELETE journal because it allows concurrent readers during a write, and it does not corrupt the database if the process crashes mid-write — relevant during development with frequent restarts.

---

## Data Flow Summary

```
Login
  → load profile from SQLite

Question asked
  → classify level (DistilBERT or heuristic)
  → detect topic (TF-IDF alias + cosine)
  → retrieve examples (sentence-transformers dense)
  → read memory hint (mastery, weak_areas, used_explanations)
  → select teaching mode (remedial / clarification / advance / default)
  → generate answer (Groq LLM, prompt assembled from all above)
  → record used explanation tag in profile (in memory)

Follow-up answered
  → evaluate response (Groq LLM → Pydantic EvaluationResult)
  → update mastery (diminishing returns formula)
  → record weak concepts
  → save profile to SQLite
```

---

## Honest Limitations

| Area | Limitation |
|------|-----------|
| **Classifier training data** | The fine-tuned DistilBERT was trained on a purpose-built dataset. Questions phrased very differently from the training distribution may be misclassified. |
| **Single-user profile model** | The profile is a flat JSON blob. There is no forgetting mechanism — weak areas accumulate indefinitely and are never auto-cleared even if mastery recovers. |
| **Retrieval grounding** | Retrieved examples are inserted into the prompt as "supporting context" but the LLM is not strictly constrained to them. Hallucinations remain possible for niche topics not well-covered in the dataset. |
| **Teaching mode heuristic** | Mode selection is driven by string-matching on the evaluator hint. It is not a trained classifier and can be confused by ambiguous LLM outputs. |
| **No session isolation** | All questions within a session share the same profile state in memory. Rapid question-answer pairs could write conflicting mastery updates if the evaluator fires out of order (unlikely in a single-user Gradio app, but worth noting). |
| **LLM cost** | Every question + every evaluation incurs a Groq API call. High-volume use requires rate-limit handling beyond the current retry logic. |

---

## Evaluation

See [`eval/run_evaluation.py`](eval/run_evaluation.py) for the personalization evaluation script. It runs 10 representative ML questions through both an empty profile (baseline) and a simulated 8-session learner profile (personalized), capturing detected teaching mode and answer excerpts for each. Results are written to [`eval/evaluation_results.md`](eval/evaluation_results.md).

The key observable signal: when a topic has a low mastery score and recorded weak concepts, the teaching mode shifts from `default` to `remedial` or `clarification`, and the LLM prompt explicitly instructs the model to address those weak areas — producing a measurably different and more targeted explanation.
