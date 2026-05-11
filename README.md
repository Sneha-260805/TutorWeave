# EduAgent

EduAgent is an adaptive AI tutor for AI/ML learning, built as a complete course project showcasing a full personalization loop. It combines fine-tuned difficulty classification, two-pass personalized RAG, semantic topic detection, a multi-mode tutor agent, Pydantic-validated evaluation, and mastery-based learner memory — all wired into a polished dark Gradio dashboard.

The adaptive loop:

1. A learner asks an AI/ML question.
2. Fine-tuned DistilBERT predicts difficulty level (97.92% test accuracy).
3. Semantic topic detection maps the question to the closest dataset topic.
4. Two-pass RAG retrieves grounding examples: Pass 1 (query-driven, top-3) + Pass 2 (weak-area targeted, top-2).
5. The Memory Agent builds a personalized hint from mastery scores, weak areas, and prior explanation styles.
6. The Tutor Agent selects a teaching mode and explains the concept adaptively.
7. The Evaluator Agent asks a conceptual follow-up question.
8. The learner answers; the Evaluator returns a Pydantic-validated `EvaluationResult`.
9. The Memory Agent updates mastery (diminishing returns), weak areas, and used-explanation tags.
10. The dashboard refreshes charts and learner insights.

A full write-up of the system design, model details, and experimental results is in [`report.tex`](report.tex).

---

## Key Features

- Login, signup, logout, and authenticated learner sessions
- Email validation and PBKDF2-SHA256 password hashing (200,000 PBKDF2 iterations, 16-byte `os.urandom` salt, `hmac.compare_digest` for timing safety)
- Per-user learner profile persistence with SQLite (WAL journal mode)
- **DistilBERT difficulty classifier** (`distilbert-base-uncased`, fine-tuned on 2,400 synthetic samples) — **97.92% test accuracy**, +35.8 pp over the keyword heuristic baseline
- Keyword heuristic fallback classifier when the model directory is absent
- **Two-tier semantic topic detection**: alias table fast path (handles abbreviations like `mlp`, `cnn`, `rag`, `lstm`) + `all-MiniLM-L6-v2` cosine similarity fallback
- **Two-pass personalized RAG**:
  - Pass 1 — query-driven top-3 from `(level, topic)` slice
  - Pass 2 — weak-area targeted top-2 using composite query `"concept + topic"`, deduplicated against Pass 1
- Per `(level, topic)` index cache — dataset encoded once, not on every query
- **Four adaptive teaching modes**: `remedial`, `clarification`, `advance`, `default` — selected from evaluator hint (priority 1) or mastery + weak areas (priority 2)
- `used_explanations` tracking per topic — prevents the LLM from repeating the same explanation style
- **Diminishing-returns mastery model**: `m += 0.20 × (1 − m)` on good, `−0.06` on partial, `−0.18` on poor; initialized at 0.5, clamped to [0, 1]
- Groq-powered Tutor Agent (`llama-3.3-70b-versatile`, temperature 0.25) for mode-specific explanations
- Voice interaction support via `voice/` module: speech-to-text, text-to-speech, and audio-enabled learner conversation
- Model setup helper script `setup_model.py` plus bundled DistilBERT model assets under `models/distilbert_eduagent_v2`
- Pydantic-validated `EvaluationResult` — `understanding_level: Literal["good","partial","poor"]`
- Shared LLM client with timeout, retry (0.75 s / 1.5 s backoff, 3 attempts), and graceful local fallback
- Dark theme Gradio UI with tabbed learner dashboard and Matplotlib progress charts
- System Insights / Research panel for demo and professor-facing explainability
- **49 unit tests** across auth, memory, and profile layers (`tests/`)
- Personalization evaluation script (`eval/run_evaluation.py`) comparing baseline vs. personalized pipeline

---

## Classifier Performance

| Method | Test Accuracy |
|--------|--------------|
| Keyword heuristic (baseline) | 62.1% |
| Fine-tuned DistilBERT (ours) | **97.92%** |

Training set: 1,920 samples · Validation: 240 · Test: 240  
Dataset: 2,400 balanced synthetic AI/ML Q&A pairs (800 per level)

---

## Architecture

```text
EduAgent/
  gradio_app.py
  report.tex                        ← conference-style research paper
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
    embedder.py                     ← shared all-MiniLM-L6-v2 singleton
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
    distilbert_eduagent_v2/         ← fine-tuned DistilBERT classifier
  setup_model.py                    ← helper script to bootstrap model assets and tokenizer
  voice/                            ← speech-to-text / text-to-speech voice interface support
  datasets/
    eduagent_dataset.csv
    eduagent_training_ready.csv
  eval/
    run_evaluation.py               ← baseline vs. personalized comparison
    evaluation_results.md           ← generated report
  tests/
    test_auth.py
    test_memory.py
    test_profile.py
  runtime/
    eduagent_app.db                 ← SQLite database (WAL mode)
  requirements.txt
```

For a detailed agent flow, design decisions, ablation results, and limitations see [`report.tex`](report.tex).

---

## File Responsibilities

### `gradio_app.py`

Imports `create_app()` from `app/main.py` and launches Gradio with custom CSS.

### `app/ui.py`

UI layout layer — dark CSS, login/signup page, two-column chat workspace, and all dashboard tab components. No backend logic.

### `app/main.py`

Orchestration and Gradio callback layer.

- Initializes the database
- Handles signup, login, logout
- Handles main learner questions — calls Tutor Agent, records used explanation, updates profile
- Calls Evaluator Agent on follow-up answers
- Formats profile display, evaluation display, charts, and System Insights
- Wires backend outputs into UI components via `_pack()` — validated at runtime, mismatches raise `ValueError` immediately

### `agents/llm_client.py`

Shared LLM client. Groq chat completions with `LLM_TIMEOUT_SECONDS=8.0`, `LLM_MAX_RETRIES=2` (3 total attempts), backoff at 0.75 s and 1.5 s, and a fallback string on persistent failure.

### `agents/tutor_agent.py`

The Tutor Agent.

- Classifies question level, detects topic, runs two-pass RAG
- Reads memory hint and evaluator strategy hint from the learner profile
- Selects teaching mode (`remedial` / `clarification` / `advance` / `default`)
- Assembles LLM prompt with mode-specific instruction block (temperature 0.25)
- Returns `(level, confidence, topic, examples_df, weak_examples_df, answer, teaching_mode)`

**Teaching modes:**

| Mode | When triggered | Behaviour |
|------|----------------|-----------|
| `remedial` | Poor mastery (< 0.35) or evaluator recommends revisit | Simpler language, concrete real-world analogy, one key concept |
| `clarification` | Partial mastery or evaluator detects confusion | Addresses known weak concepts directly, builds on what is known |
| `advance` | High mastery (≥ 0.75) and no weak areas | Advanced edge cases, nuances, connections to related topics |
| `default` | All other cases | Balanced level-appropriate explanation |

### `agents/evaluator_agent.py`

The Evaluator Agent.

- Generates one follow-up question (temperature 0.2)
- Evaluates learner response with Pydantic-validated `EvaluationResult` (temperature 0.0)
- `_extract_json_object()` strips markdown fences and uses a `\{.*\}` DOTALL regex fallback
- Safe fallback on parse failure: `understanding_level="partial"` (not `"poor"`) to avoid incorrectly triggering remedial mode

### `agents/memory_agent.py`

The Memory Agent. Owns the learner profile schema and all mutations.

- `ensure_profile_structure` normalizes profile shape on load
- Mastery update with **diminishing returns**: `m += 0.20 × (1 − m)` on good; `m -= 0.06` on partial; `m -= 0.18` on poor; clamped to [0, 1]; initialized at 0.5
- `record_used_explanation(topic, f"{level}-{mode}")` tags explanation style per topic
- `build_memory_hint()` — up to 5 conditional sentences surfacing mastery, weak areas, used styles, topic history
- `build_evaluation_strategy_hint()` — maps last `EvaluationResult` to a mode string + action addendum for the tutor

### `ml/embedder.py`

Shared singleton loader for `all-MiniLM-L6-v2`. Loaded once at import time; `semantic_available` flag used by both retriever and topic detector to choose backend.

### `ml/classifier.py`

- Loads fine-tuned DistilBERT from `models/distilbert_eduagent_v2/`
- Falls back to keyword heuristic if model directory is absent (62.1% accuracy vs. 97.92% for DistilBERT)
- Beginner-intent calibration applies only to questions of ≤ 8 words
- **HuggingFace Hub**: Model also published at [`SSneha2005/Eduagent_distilbert`](https://huggingface.co/SSneha2005/Eduagent_distilbert) for direct loading via `transformers`

### `ml/topic_detector.py`

Two-tier topic detection:

1. **Alias table** — space-padded exact substring match against 14 canonical topics and their abbreviations/synonyms (e.g. `mlp`, `ann`, `rag`, `lstm`, `sgd`). Authoritative — skips embedding step when matched.
2. **Semantic similarity** — `all-MiniLM-L6-v2` encodes `"{topic}: {all questions for topic}"` per topic per level, cached per dataset snapshot. Cosine similarity ranks best match.
3. TF-IDF fallback (ngram 1–2) if `sentence-transformers` is unavailable.

### `ml/retriever.py`

Two-pass personalized RAG:

- **Pass 1** — query-driven top-`RAG_TOP_N` (default 3) from `(level, topic)` slice
- **Pass 2** — weak-area targeted top-`RAG_WEAK_TOP_N` (default 2) via composite query `"concept topic"`, pooled across weak concepts, deduplicated against Pass 1
- Answer length bounds filter per level; complexity penalty (+0.15 per hard term, +0.15 if > 18 words) discourages mismatched examples
- `(len(df), level, topic)` cache key — index built once per dataset/level/topic combination

### `auth/auth_service.py`

Signup with email format validation and password length check. Login by email or username.

### `auth/password_utils.py`

PBKDF2-SHA256 via `passlib`. `hmac.compare_digest` for timing-safe verification.

### `db/sqlite_store.py`

Opens SQLite connections with `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`. Initializes `users` and `profiles` tables.

### `db/profile_repository.py`

Creates, loads, and saves per-user learner profiles as JSON blobs. `INSERT OR IGNORE` with rollback guard on profile creation.

### `eval/run_evaluation.py`

Runs 10 representative questions through the pipeline twice each — blank profile (baseline) vs. simulated 8-session profile (personalized). Captures level, confidence, topic, teaching mode, and 55-word answer excerpt per run. Writes `eval/evaluation_results.md`.

```powershell
python -m eval.run_evaluation
```

### `tests/`

**49 unit tests** across three files:

| File | What it tests |
|------|--------------|
| `test_profile.py` | `ensure_profile_structure` — defaults, idempotency, type coercion, round-trips |
| `test_memory.py` | Mastery scoring, diminishing returns, `used_explanations`, memory hints |
| `test_auth.py` | Password hashing, email validation, `register_user` (mocked DB) |

```powershell
python -m pytest tests/ -v
```

---

## Learner Profile Schema

| Field | Type | Description |
|-------|------|-------------|
| `sessions` | int | Login count |
| `questions_asked` | int | Total questions across all sessions |
| `last_level` | str | Most recently detected level |
| `level_history` | list | Per-question level sequence |
| `topics_seen` | list | Ordered first-seen topic list |
| `topic_counts` | dict | Questions asked per topic |
| `mastery` | dict | 0.0–1.0 score per topic (initialized 0.5) |
| `weak_areas` | dict | Concept-level gaps per topic from evaluator |
| `used_explanations` | dict | `level-mode` tags used per topic |
| `recommended_next_topics` | list | Evaluator-suggested next topics |
| `last_evaluation` | dict | Most recent `EvaluationResult` |

---

## End-to-End Runtime Flow

### 1. Authentication

Signup (email validated, password hashed with PBKDF2-SHA256) or login. Learner profile loaded or created in SQLite.

### 2. Classification

Fine-tuned DistilBERT predicts `beginner` / `intermediate` / `advanced` with confidence scores (97.92% test accuracy). Keyword heuristic fires automatically if model is absent.

### 3. Semantic Topic Detection

Alias table check (fast path) → semantic cosine similarity over per-level topic index (all-MiniLM-L6-v2) → TF-IDF fallback.

### 4. Two-Pass RAG

Pass 1 retrieves top-3 query-similar examples from the `(level, topic)` slice. Pass 2 retrieves top-2 examples targeting recorded weak concepts, deduplicated against Pass 1.

### 5. Memory Read

Memory Agent builds personalized hint: mastery score, weak concepts, previously used explanation styles for the detected topic.

### 6. Teaching Mode Selection

`infer_teaching_mode()` maps evaluator hint (priority 1) or mastery + weak areas (priority 2) to `remedial` / `clarification` / `advance` / `default`.

### 7. Tutoring

Tutor Agent assembles LLM prompt with mode-specific instruction block. Calls Groq (`llama-3.3-70b-versatile`, temperature 0.25). Local template fallback on timeout.

### 8. Explanation Tag Recorded

`record_used_explanation(topic, f"{level}-{teaching_mode}")` ensures the same style is not repeated on the next visit.

### 9. Follow-up and Evaluation

Evaluator Agent generates a follow-up question (temperature 0.2), then evaluates the learner's answer (temperature 0.0) and returns a Pydantic-validated `EvaluationResult`.

### 10. Memory Update

Mastery updated (diminishing returns), weak concepts recorded, profile saved to SQLite.

### 11. Dashboard Update

UI refreshes: learner snapshot, memory summary, evaluation result, Mastery by Topic chart, Topic Revisit Count chart, Weak Concept Count chart, System Insights.

---

## Dataset Requirements

`datasets/eduagent_dataset.csv` with columns:

```text
question, answer, level, topic
```

Training dataset (`eduagent_training_ready.csv`): 2,400 balanced synthetic AI/ML Q&A pairs (800 per level — beginner, intermediate, advanced) across 14 topics.

---

## Model Requirements

Fine-tuned DistilBERT classifier at:

```text
models/distilbert_eduagent_v2/
  config.json
  model.safetensors
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
  README.md                             ← model card with training details
```

The model weights are also published on the HuggingFace Hub for direct loading without cloning the repository:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_id = "Sneha-260805/distilbert-eduagent-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
```

**Hub link**: [`Sneha-260805/distilbert-eduagent-v2`](https://huggingface.co/Sneha-260805/distilbert-eduagent-v2)

For training details, performance metrics, and usage examples, see [`models/distilbert_eduagent_v2/README.md`](models/distilbert_eduagent_v2/README.md).

If the model directory is absent, the keyword heuristic activates automatically (functional but lower accuracy: 62.1% vs. 97.92%).

---

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
LLM_TIMEOUT_SECONDS=8
LLM_MAX_RETRIES=2
RAG_TOP_N=3
RAG_WEAK_TOP_N=2
```

Only `GROQ_API_KEY` is required.

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

Key dependencies: `transformers`, `torch`, `sentence-transformers`, `gradio`, `groq`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `python-dotenv`, `passlib`, `pydantic`.

### 4. Configure Environment

Add `GROQ_API_KEY` to `.env` (see above).

### 5. Run

```powershell
python .\gradio_app.py
```

Open `http://127.0.0.1:7860` in your browser.

---

## Validation

Build check:

```powershell
python -c "from app.main import create_app; app = create_app(); print('ok')"
```

Unit tests (49 tests):

```powershell
python -m pytest tests/ -v
```

Personalization evaluation (requires `GROQ_API_KEY`):

```powershell
python -m eval.run_evaluation
```

Generates `eval/evaluation_results.md` — side-by-side comparison of blank-profile vs. personalized tutor behavior across 10 questions.

---

## Common Issues

| Issue | Resolution |
|-------|------------|
| Missing `GROQ_API_KEY` | Add to `.env` or set `$env:GROQ_API_KEY` in shell |
| Dataset not found | Ensure `datasets/eduagent_dataset.csv` exists |
| Classifier files missing | Keyword heuristic activates automatically — classification still works |
| Empty progress charts | Normal for new profiles — populate after a few questions and evaluations |
| No follow-up context | Ask a main question first, then answer the follow-up in the Evaluate tab |
| Login/profile issues | Restart app so `init_db()` re-initializes the SQLite tables |

---

## Project Strengths

- **97.92% classifier accuracy** — +35.8 pp over keyword heuristic, zero GPU required at inference (CPU DistilBERT)
- **Two-pass personalized RAG** — separate query-driven and weak-area retrieval passes ground every answer in relevant dataset context
- **Semantic topic detection** — alias table handles common abbreviations; MiniLM-L6-v2 handles paraphrases
- **Four teaching modes** with different LLM instruction blocks — measurably different explanation styles for remedial vs. advanced learners
- **Diminishing-returns mastery** — mastery cannot trivially reach 1.0; gains compress near saturation
- **`used_explanations` deduplication** — same explanation style never repeated for a topic
- **Pydantic-validated evaluator** — no unsafe raw dict access from LLM JSON responses
- **49 unit tests** covering auth, memory, and profile layers
- **SQLite WAL journal** — no database corruption on crash
- **Local fallback answer** — app never hard-crashes on LLM timeout
- **System Insights panel** for demo and professor-facing explainability
- Conference-style research paper [`report.tex`](report.tex) included

---

## Limitations

- Mastery is a heuristic score, not a validated pedagogical model (e.g. Bayesian Knowledge Tracing).
- Weak areas accumulate indefinitely — no forgetting or auto-clearing when mastery recovers.
- Teaching mode selection uses string-matching on the LLM evaluator hint, not a trained classifier.
- Retrieved examples ground the prompt but do not strictly prevent hallucinations for niche topics.
- Single-user Gradio app — not designed for concurrent multi-user production deployment.

---

## Future Work

- Replace heuristic mastery with Bayesian Knowledge Tracing
- Weak area decay / auto-clearing when mastery crosses a recovery threshold
- Per-topic learning paths and exportable learner reports
- Instructor analytics dashboard
- Automated LLM response quality evaluation (G-Eval / LLM-as-judge)
- Evaluation history charts
- Production frontend with multi-user support

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Gradio 6.0 |
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Classifier | DistilBERT (`distilbert-base-uncased`, fine-tuned) via HuggingFace Transformers |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) |
| TF-IDF fallback | scikit-learn `TfidfVectorizer` |
| Database | SQLite (WAL journal) |
| Auth | `passlib` PBKDF2-SHA256 |
| Validation | Pydantic v2 |
| Charts | Matplotlib |
| Data | Pandas, NumPy, scikit-learn |

---

## Summary

EduAgent demonstrates a complete adaptive learning loop:

```
Question → DistilBERT Classify → Semantic Topic Detection
→ Two-Pass RAG (query-driven + weak-area) → Read Memory
→ Select Teaching Mode → Tutor Answer (mode-specific)
→ Follow-up Question → Evaluate (Pydantic-validated)
→ Update Mastery (diminishing returns) + Weak Areas + Used Explanations
→ Save Profile → Dashboard Refresh
```

Explanation style, depth, and focus change measurably between a first-time learner and a returning learner with recorded mastery history — the core goal of the adaptive personalization design.
