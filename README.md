# EduAgent

EduAgent is an adaptive AI tutor for AI/ML learning. It is designed as a complete academic showcase project, not just a simple chatbot. The app combines authentication, difficulty classification, semantic retrieval, topic detection, a multi-mode tutor agent, follow-up evaluation, learner memory with mastery tracking, progress charts, and a polished dark Gradio dashboard.

The main goal is to demonstrate a full adaptive learning loop:

1. A learner asks an AI/ML question.
2. EduAgent predicts the learner's difficulty level (fine-tuned DistilBERT).
3. EduAgent detects the topic and retrieves semantically relevant dataset examples.
4. The Memory Agent reads the learner's mastery, weak areas, and prior explanation styles.
5. The Tutor Agent selects a teaching mode and explains the concept adaptively.
6. The Evaluator Agent asks a follow-up question.
7. The learner answers the follow-up.
8. The Memory Agent updates mastery, weak areas, topic history, and used-explanation tags.
9. The dashboard updates the learner profile and charts.

---

## Key Features

- Login, signup, logout, and authenticated learner sessions
- Email validation and PBKDF2-SHA256 password hashing on signup
- Per-user learner profile persistence with SQLite (WAL journal mode)
- Difficulty classification into beginner, intermediate, and advanced levels (fine-tuned DistilBERT)
- Keyword heuristic fallback classifier when the model directory is absent
- Topic detection with alias expansion (e.g. `llm → large language models`)
- **Semantic retrieval** using `sentence-transformers` (`all-MiniLM-L6-v2`) dense embeddings with TF-IDF fallback
- Per `(level, topic)` index cache — questions encoded once, not on every query
- **Four adaptive teaching modes**: remedial, clarification, advance, default — selected from mastery and last evaluation
- `used_explanations` tracking per topic — prevents the LLM from repeating the same explanation style
- Mastery scoring with **diminishing returns**: gains shrink as mastery approaches 1.0
- Groq-powered Tutor Agent (`llama-3.3-70b-versatile`) for level-aware, mode-specific explanations
- Evaluator Agent for follow-up questions and Pydantic-validated understanding checks
- Shared LLM client with timeout, retry, and graceful local fallback
- Memory Agent for learner state, weak areas, mastery, and recommendations
- Dark theme Gradio UI designed for a project demo or professor presentation
- Two-column workspace with a chat area and organized learner dashboard
- Tabbed dashboard sections to avoid clutter
- Matplotlib progress charts with dark theme styling
- System Insights / Research panel for showing internal AI pipeline signals
- **45 unit tests** across auth, memory, and profile layers (`tests/`)
- **Personalization evaluation script** (`eval/run_evaluation.py`) with markdown report

---

## Reliability and Architecture Improvements

All earlier prototype weaknesses have been addressed:

- **Semantic retrieval replaces TF-IDF** for example lookup. `sentence-transformers` (`all-MiniLM-L6-v2`) dense embeddings are used as the primary backend. TF-IDF is retained as an automatic fallback if the package is absent.
- **Classifier path corrected** to `models/distilbert_eduagent_v2/`. The old `difficulty_classifier/` placeholder path is no longer referenced.
- **Beginner heuristic gated** to questions of ≤ 8 words — longer questions no longer get incorrectly downgraded.
- **Mastery uses diminishing returns**: `mastery += 0.20 × (1 − mastery)` on a good evaluation, so mastery can no longer trivially reach 1.0.
- **`used_explanations` now tracked**: `record_used_explanation` is called after every tutor response, tagging the topic with the `level-mode` key used (e.g. `intermediate-advance`). The memory hint surfaces these tags so the LLM avoids repeating the same approach.
- **Teaching mode returned** from `generate_tutor_response` and wired into the memory system.
- **`_pack()` output validation**: all Gradio handler return tuples are built from named dicts and validated against a key list at runtime — a mismatch raises `ValueError` immediately instead of silently routing values to wrong components.
- **Auth improvements**: email validation on signup, auto tab-switch from Signup → Login on success, login identifier pre-filled with registered email.
- **SQLite journal mode changed** from `MEMORY` to `WAL` — prevents database corruption on crash.
- **MongoDB placeholder removed** — `pymongo` dependency and dead connection code fully deleted.
- **Dead UI code removed** — the old `build_ui()` function (~120 lines) and module-level handler aliases were cleaned up.
- **Unit tests added**: 45 tests across `tests/test_auth.py`, `tests/test_memory.py`, and `tests/test_profile.py`.

---

## Current UI

EduAgent uses a dark, presentation-ready Gradio interface.

### Login Page

- Dark academic showcase hero section
- Login and signup tabs (auto-switches to Login on successful signup)
- Clean authentication card
- Project capability highlights

### Chat Page

Left column:

- Tutor conversation
- Main question input
- Ask EduAgent button (shows "Thinking…" and disables during inference)
- Clear Chat button

Right column — tabbed learner dashboard:

- `Overview` — detected level, topic, confidence, learner memory summary
- `Evaluate` — follow-up question, answer input, evaluation result
- `Progress` — Mastery by Topic, Topic Revisit Count, Weak Concept Count charts
- `Research` — System Insights panel with internal pipeline signals and retrieved examples

---

## System Insights

Placed under the `Research` tab for demos and professor-facing explanations. Displays:

- Predicted level and confidence scores
- Detected topic
- Memory hint used by the Tutor Agent (mastery, weak areas, used explanations)
- Evaluator strategy hint and teaching mode selected
- Last evaluation summary and JSON
- Retrieved examples from the dataset

---

## Architecture

For a detailed agent flow diagram, component breakdown, design decisions, and honest limitations, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

```text
EduAgent/
  gradio_app.py
  ARCHITECTURE.md
  app/
    main.py
    ui.py
  agents/
    llm_client.py
    tutor_agent.py
    evaluator_agent.py
    memory_agent.py
  ml/
    classifier.py
    topic_detector.py
    retriever.py
  auth/
    auth_service.py
    password_utils.py
  db/
    sqlite_store.py
    profile_repository.py
  config/
    settings.py
  models/
    distilbert_eduagent_v2/      ← fine-tuned DistilBERT classifier
  datasets/
    eduagent_dataset.csv
    eduagent_training_ready.csv
  eval/
    run_evaluation.py            ← baseline vs. personalized comparison
    evaluation_results.md        ← generated report
  tests/
    test_auth.py
    test_memory.py
    test_profile.py
  runtime/
    eduagent_app.db              ← SQLite database (WAL mode)
  requirements.txt
```

---

## File Responsibilities

### `gradio_app.py`

Imports `create_app()` from `app/main.py` and launches the Gradio app with custom CSS.

### `app/ui.py`

UI layout layer. Defines the custom dark CSS, login/signup page, two-column chat workspace, and all dashboard tab components. Contains no backend logic.

### `app/main.py`

Orchestration and Gradio callback layer.

- Initializes the database
- Handles signup, login, and logout
- Handles main learner questions — calls Tutor Agent, records used explanation, updates profile
- Calls Evaluator Agent on follow-up answers
- Formats profile display, evaluation display, charts, and System Insights
- Wires backend outputs into UI components via `_pack()` for validated tuple construction

### `agents/llm_client.py`

Shared LLM client. Calls Groq chat completions with timeout, retries, logging, and fallback text.

### `agents/tutor_agent.py`

The Tutor Agent.

- Predicts difficulty level, detects topic, retrieves semantic examples
- Reads memory hints and evaluator strategy hints from the learner profile
- Selects teaching mode (`remedial` / `clarification` / `advance` / `default`)
- Assembles and fires the LLM prompt with mode-specific instructions
- Returns `(level, confidence, topic, examples_df, answer, teaching_mode)`

Teaching mode is driven by mastery score, weak areas, and the evaluator's `recommended_action`. Each mode injects a different instruction block into the prompt, producing measurably different explanation styles.

### `agents/evaluator_agent.py`

The Evaluator Agent.

- Generates one follow-up question after the tutor answer
- Evaluates the learner's response
- Validates evaluator JSON with Pydantic (`EvaluationResult` model)
- Returns `understanding_level` (`good` / `partial` / `poor`), `weak_concepts`, `feedback`, `recommended_action`

### `agents/memory_agent.py`

The Memory Agent. Owns the learner profile schema and all mutations.

- Normalizes profile shape on load (`ensure_profile_structure`)
- Tracks sessions, questions asked, topics seen, topic counts, level history
- Updates mastery with diminishing returns: `mastery += 0.20 × (1 − mastery)` on good; `−0.06` on partial; `−0.18` on poor
- Records weak concepts per topic
- Tracks `used_explanations` per topic to prevent repeated explanation styles
- Builds memory hints and evaluator strategy hints for the tutor prompt

### `ml/classifier.py`

- Loads fine-tuned DistilBERT from `models/distilbert_eduagent_v2/`
- Falls back to a keyword heuristic if the model directory is missing
- Beginner-intent calibration applies only to questions of ≤ 8 words
- Returns predicted level and confidence scores

### `ml/topic_detector.py`

- Cleans and normalizes question text
- Alias expansion table maps shortforms to full topic names
- TF-IDF cosine similarity over dataset topic strings
- Cached per-dataset index

### `ml/retriever.py`

- **Primary backend**: `sentence-transformers` (`all-MiniLM-L6-v2`) dense embeddings
- **Fallback backend**: TF-IDF cosine similarity (sklearn) — activates automatically if `sentence-transformers` is not installed
- Dataset slice filtered by `(level, topic)` — encoded once and cached in `_INDEX_CACHE`
- Complexity penalty discourages mismatched advanced examples for simple questions
- Returns top-N `(question, answer, level, topic)` rows as grounding context for the tutor prompt

### `auth/auth_service.py`

- Signup with email format validation (`_is_valid_email`) and password length check
- Login with email or username lookup
- User creation through `db/profile_repository.py`

### `auth/password_utils.py`

- PBKDF2-SHA256 hashing via `passlib`
- Timing-safe comparison on verify

### `db/sqlite_store.py`

- Opens SQLite connections with `PRAGMA journal_mode=WAL`
- Initializes `users` and `profiles` tables

### `db/profile_repository.py`

- Creates, loads, and saves per-user learner profiles as JSON blobs
- `INSERT OR IGNORE` with rollback guard on profile creation to handle race conditions

### `eval/run_evaluation.py`

Personalization evaluation script.

- Runs 10 representative ML/AI questions through the full pipeline twice each: once with an empty profile (baseline) and once with a simulated 8-session learner profile (personalized — with mastery scores, weak areas, and `used_explanations` populated)
- Captures detected level, confidence, topic, teaching mode, and a 55-word answer excerpt per run
- Writes `eval/evaluation_results.md` with a per-question comparison table and summary counts

Run with:

```powershell
python -m eval.run_evaluation
```

### `tests/`

45 unit tests across three files:

| File | What it tests |
|------|--------------|
| `test_profile.py` | `ensure_profile_structure` — defaults, idempotency, type coercion, round-trips |
| `test_memory.py` | Mastery scoring, diminishing returns, used_explanations, memory hints |
| `test_auth.py` | Password hashing, email validation, `register_user` (with mocked DB) |

Run all tests:

```powershell
python -m pytest tests/ -v
```

---

## Learner Profile Schema

Each learner profile stores:

| Field | Type | Description |
|-------|------|-------------|
| `sessions` | int | Login count |
| `questions_asked` | int | Total questions in all sessions |
| `last_level` | str | Most recently detected level |
| `level_history` | list | Per-question level sequence |
| `topics_seen` | list | Ordered first-seen topic list |
| `topic_counts` | dict | Questions asked per topic |
| `mastery` | dict | 0.0–1.0 score per topic (starts 0.5) |
| `weak_areas` | dict | Concept-level gaps per topic from evaluator |
| `used_explanations` | dict | `level-mode` tags used per topic |
| `recommended_next_topics` | list | Evaluator-suggested next topics |
| `last_evaluation` | dict | Most recent `EvaluationResult` |

---

## End-to-End Runtime Flow

### 1. Authentication

User signs up (email validated, password hashed) or logs in. EduAgent loads or creates a learner profile from SQLite.

### 2. Question Asking

The learner asks an AI/ML question.

### 3. Classification

Fine-tuned DistilBERT predicts level (`beginner` / `intermediate` / `advanced`) with confidence scores. Heuristic fallback fires if the model is absent.

### 4. Topic Detection

Alias expansion + TF-IDF cosine over dataset topic strings returns the best matching topic.

### 5. Retrieval

`sentence-transformers` dense embeddings retrieve the top-2 semantically similar Q&A pairs from the dataset, filtered to the detected `(level, topic)` slice.

### 6. Memory Read

The Memory Agent reads mastery score, weak concepts, and previously used explanation styles for the detected topic, and builds the memory hint and evaluator strategy hint.

### 7. Teaching Mode Selection

`infer_teaching_mode()` maps the evaluator strategy hint to one of four modes: `remedial`, `clarification`, `advance`, or `default`. Each mode injects a different instruction block into the tutor prompt.

### 8. Tutoring

The Tutor Agent calls Groq (`llama-3.3-70b-versatile`, temperature 0.25) with the assembled prompt. A local template fallback is available if the API times out.

### 9. Explanation Tag Recorded

`record_used_explanation(topic, f"{level}-{teaching_mode}")` tags the topic so the same style is not repeated on the next visit.

### 10. Follow-up Question

The Evaluator Agent generates one short conceptual follow-up question.

### 11. Evaluation

The learner answers. The Evaluator Agent scores the response and returns a Pydantic-validated `EvaluationResult`.

### 12. Memory Update

The Memory Agent updates mastery (diminishing returns), records weak concepts, and saves the profile to SQLite.

### 13. Dashboard Update

The UI refreshes: learner snapshot, memory summary, evaluation result, charts, system insights.

---

## Dataset Requirements

EduAgent expects `datasets/eduagent_dataset.csv`.

Required columns:

```text
question, answer, level, topic
```

Example row:

```csv
question,answer,level,topic
"What is supervised learning?","Supervised learning is...",beginner,Machine Learning
```

---

## Model Requirements

EduAgent expects the fine-tuned DistilBERT classifier at:

```text
models/distilbert_eduagent_v2/
```

Typical files:

```text
config.json
model.safetensors
tokenizer_config.json
special_tokens_map.json
vocab.txt
```

If this directory is absent, the keyword heuristic classifier activates automatically.

---

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
```

Only `GROQ_API_KEY` is required. The other variables have sensible defaults.

You can also set the API key in PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

---

## Setup

### 1. Clone

```powershell
git clone https://github.com/Sneha-260805/EduAgent.git
cd EduAgent
```

### 2. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `scikit-learn`, `transformers`, `torch`, `sentence-transformers`, `gradio`, `groq`, `matplotlib`, `seaborn`, `python-dotenv`, `passlib`, `pydantic`.

### 4. Configure Environment

```text
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the App

```powershell
python .\gradio_app.py
```

Open the local URL printed by Gradio (usually `http://127.0.0.1:7860`).

---

## Validation

Build check:

```powershell
python -c "from app.main import create_app; app = create_app(); print('app build ok')"
```

Unit tests:

```powershell
python -m pytest tests/ -v
```

Personalization evaluation (requires `GROQ_API_KEY`):

```powershell
python -m eval.run_evaluation
```

This generates `eval/evaluation_results.md` — a side-by-side comparison of tutor behavior with and without personalization active.

---

## Common Issues

### Missing `GROQ_API_KEY`

Make sure `.env` exists or set the variable in the current shell.

### Dataset Not Found

Make sure `datasets/eduagent_dataset.csv` exists.

### Classifier Files Not Found

If `models/distilbert_eduagent_v2/` is absent the keyword heuristic classifier activates automatically. Classification still works, but with lower accuracy on ambiguous phrasing.

### Empty Charts

Normal for a new learner profile. Charts populate after questions and follow-up evaluations.

### No Follow-up Context

Ask a main question first, then answer the generated follow-up in the Evaluate tab.

### Login or Profile Issues

Restart the app so `init_db()` runs and initializes the required SQLite tables.

---

## Project Strengths

- Full adaptive learning loop — classify, retrieve, tune, evaluate, remember, repeat
- Semantic retrieval with dense embeddings, not just keyword matching
- Four adaptive teaching modes driven by real learner history
- Diminishing-returns mastery model — cannot trivially max out
- `used_explanations` prevents the LLM from repeating the same explanation style
- Pydantic-validated evaluator output — no raw dict access from LLM responses
- Validated Gradio handler tuples — output routing errors caught at runtime
- 45 unit tests covering auth, memory, and profile layers
- SQLite WAL journal — no database corruption on crash
- Local fallback answer — app never hard-crashes on LLM timeout
- System Insights panel for explainability during demos

---

## Current Limitations

- Mastery is a heuristic score, not a validated pedagogical model (e.g. Bayesian Knowledge Tracing).
- Weak areas accumulate indefinitely — there is no forgetting or auto-clearing mechanism when mastery recovers.
- Teaching mode selection is driven by string-matching on the LLM evaluator hint, not a trained classifier.
- Retrieved examples ground the tutor prompt but do not strictly prevent hallucinations for niche topics.
- Single-user Gradio app — not designed for concurrent multi-user production deployment.

---

## Future Improvements

- Replace heuristic mastery with Bayesian Knowledge Tracing
- Add a forgetting/decay mechanism to weak areas
- Add per-topic learning paths and exportable learner reports
- Add instructor/admin analytics dashboard
- Add automated LLM response quality evaluation (e.g. G-Eval)
- Add evaluation history charts
- Add a custom frontend for production deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6.0 |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Classifier | DistilBERT (fine-tuned) via HuggingFace Transformers |
| Retrieval | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Database | SQLite (WAL journal) |
| Auth | `passlib` PBKDF2-SHA256 |
| Validation | Pydantic |
| Charts | Matplotlib |
| Data | Pandas, scikit-learn |

---

## Summary

EduAgent is an adaptive AI/ML tutor with a complete personalization loop:

```
Question → Classify → Detect Topic → Semantic Retrieval → Read Memory
→ Select Teaching Mode → Tutor Answer → Follow-up → Evaluate
→ Update Mastery + Weak Areas + Used Explanations → Save Profile
```

It demonstrates real adaptive behavior — the explanation style, depth, and focus change measurably between a first-time learner and a returning learner with recorded history.
