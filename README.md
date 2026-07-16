# TutorWeave

TutorWeave is an adaptive AI/ML tutoring application built with Gradio. It combines a fine-tuned local DistilBERT difficulty classifier, semantic topic detection, two-pass personalized RAG, learner memory, follow-up evaluation, SQLite user profiles, and voice interaction through speech-to-text and text-to-speech.

The project is designed as a complete academic/course project: it is not just a chatbot UI. It tracks each learner, adapts explanation style based on mastery and weak areas, retrieves relevant dataset examples before answering, evaluates the learner's follow-up response, and updates the learner profile after each interaction.




---

## What The App Does

1. A learner signs up or logs in.
2. The learner asks an AI/ML question by typing or recording audio.
3. The local DistilBERT classifier predicts the difficulty level: `beginner`, `intermediate`, or `advanced`.
4. The topic detector maps the question to the closest AI/ML topic.
5. The RAG retriever pulls relevant examples from the dataset.
6. If the learner has recorded weak areas, a second RAG pass retrieves examples targeting those weak concepts.
7. The memory agent builds a personalized hint from previous sessions, mastery, weak areas, and used explanation styles.
8. The tutor agent asks the Gemini LLM to generate a level-aware, mode-aware answer grounded in retrieved examples.
9. The evaluator agent generates a follow-up question.
10. The learner answers the follow-up.
11. The evaluator returns a structured assessment: `good`, `partial`, or `poor`.
12. The memory agent updates mastery, weak areas, recommendations, and learner history.
13. The dashboard refreshes charts and learner insights.
14. The learner can listen to the tutor answer through text-to-speech.

---

## Main Features

- Login, signup, logout, and authenticated learner sessions.
- SQLite-backed user and profile storage.
- PBKDF2-SHA256 password hashing through `passlib`.
- Local fine-tuned DistilBERT classifier stored in `models/distilbert_eduagent_v2`.
- Hugging Face classifier fallback through `CLASSIFIER_HF_REPO`.
- Classifier calibration for short comparison questions, so ordinary comparisons like "compare Adam and RMSProp" are treated as `intermediate` while deeper mathematical comparisons stay `advanced`.
- Semantic topic detection with alias matching and locally cached MiniLM embeddings, plus TF-IDF fallback.
- Two-pass personalized RAG:
  - Pass 1: retrieve examples similar to the learner question.
  - Pass 2: retrieve examples targeting learner weak areas.
- Adaptive teaching modes:
  - `default`
  - `remedial`
  - `clarification`
  - `advance`
- Learner memory with sessions, topic history, mastery, weak areas, used explanation styles, and last evaluation.
- Gemini-powered tutor and evaluator agents.
- Student-friendly bullet-point tutor answers with bracket citation markers disabled.
- Structured evaluation with Pydantic validation.
- Voice input using Whisper through `openai-whisper`.
- Voice output using local text-to-speech through `pyttsx3` fallback and the `voice/` module.
- Gradio dashboard with profile summaries, retrieved examples, system insights, and progress charts.
- Unit tests for auth, memory, and profile behavior.
- RAG evaluation scripts and saved evaluation outputs.

---

## Current Important Project Choices

### Primary Classifier Model

The main classifier model is the local DistilBERT model:

```text
models\distilbert_eduagent_v2
```

The active setting is in [`config/settings.py`](config/settings.py):

```python
CLASSIFIER_PATH = str(BASE_DIR / "models" / "distilbert_eduagent_v2")
CLASSIFIER_HF_REPO = os.getenv("CLASSIFIER_HF_REPO", "SSneha2005/Eduagent_distilbert")
```

The trained DistilBERT weights are also uploaded on Hugging Face at
[`SSneha2005/Eduagent_distilbert`](https://huggingface.co/SSneha2005/Eduagent_distilbert).
TutorWeave uses this repository as the fallback source when local model weights are not available.

The local model folder contains:

```text
models/distilbert_eduagent_v2/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json
  vocab.txt
  label_map.json
  training_config.json
  README.md
  reports/
  splits/
  checkpoint-240/
  checkpoint-480/
  checkpoint-720/
```

At startup, [`ml/classifier.py`](ml/classifier.py) first checks for local weights at `CLASSIFIER_PATH`. If `model.safetensors` or `pytorch_model.bin` exists, it loads the local model. If local loading fails, it tries the Hugging Face repo in `CLASSIFIER_HF_REPO`. If both fail, it falls back to heuristics.

The classifier also applies a small calibration layer after the DistilBERT prediction:

- Very simple definition questions remain `beginner`.
- Ordinary comparison questions such as `compare Adam optimizer and RMSProp` are treated as `intermediate`.
- Comparison questions with deeper markers such as `derive`, `convergence`, `bias correction`, `mathematically`, or `loss landscape` are treated as `advanced`.
- Explicit advanced wording such as `derive`, `prove`, `architecture`, `convergence`, or `backpropagation` can override the raw model label.

Example classifications:

```text
What is gradient descent? -> beginner
Compare RMSProp and Adam -> intermediate
Compare Adam and RMSProp in terms of convergence -> advanced
Derive the Adam update rule and compare it with RMSProp -> advanced
```

### Main Dataset

The active RAG dataset is configured in [`config/settings.py`](config/settings.py):

```python
DATASET_FILE = str(BASE_DIR / "datasets" / "data_easy" / "eduagent_dataset_easy_v2.csv")
```

The dataset should contain at least:

```text
question, answer, level, topic
```

Some generated datasets may also include extra metadata columns such as `subtopic`, `question_angle`, `key_concepts`, or generation metadata.

### User Data

User data is stored in SQLite:

```text
runtime\eduagent_app.db
```

The database file can be overridden with:

```text
TUTORWEAVE_DB_FILE=path\to\custom.db
```

Important tables:

```text
users
profiles
```

The `users` table stores authentication data. The `profiles` table stores learner memory as JSON strings.

---

## Project Structure

```text
TutorWeave/
  gradio_app.py
  README.md
  ARCHITECTURE.md
  report.tex
  report.pdf
  requirements.txt
  .env.example

  app/
    main.py                 # Gradio callbacks, orchestration, auth flow, question/evaluation flow
    ui.py                   # Gradio UI layout, dashboard, voice controls, CSS

  agents/
    llm_client.py           # Gemini client, retries, quota handling, fallback behavior
    tutor_agent.py          # classification + topic detection + RAG + prompt assembly
    evaluator_agent.py      # follow-up question generation and response evaluation
    memory_agent.py         # learner profile schema and profile update logic

  ml/
    classifier.py           # local DistilBERT loading and fallback heuristics
    embedder.py             # shared local-cache all-MiniLM-L6-v2 embedder
    topic_detector.py       # alias matching, semantic topic detection, TF-IDF fallback
    retriever.py            # two-pass RAG retrieval
    prompts.py

  voice/
    __init__.py
    speech_to_text.py       # Whisper speech-to-text wrapper
    text_to_speech.py       # TTS engine wrapper
    utils.py                # Gradio-facing voice helpers

  auth/
    auth_service.py         # signup/login validation
    password_utils.py       # password hashing and verification

  db/
    sqlite_store.py         # SQLite connection and table initialization
    profile_repository.py   # users and learner profiles

  config/
    settings.py             # dataset, model, DB, LLM, and RAG settings

  datasets/
    data_easy/
      eduagent_dataset_easy_v2.csv
    eduagent_dataset.csv
    eduagent_training_ready.csv

  models/
    distilbert_eduagent_v2/ # primary local classifier model

  eval/
    run_evaluation.py
    evaluation_results.md
    rag_eval_report.md
    rag_eval_results.*

  tests/
    test_auth.py
    test_memory.py
    test_profile.py

  runtime/
    eduagent_app.db         # active SQLite app database
```

---

## Core Components

### `gradio_app.py`

Launches the Gradio app:

```python
demo = create_app()
demo.launch(theme=gr.themes.Soft(), css=CUSTOM_CSS, share=True)
```

If you do not want a public Gradio share link, remove `share=True`.

### `app/main.py`

This is the main application orchestration layer. It:

- Initializes the SQLite database.
- Handles signup, login, logout.
- Loads and saves learner profiles.
- Handles learner questions.
- Calls the tutor agent.
- Stores used explanation modes.
- Generates follow-up questions.
- Evaluates follow-up replies.
- Updates learner memory.
- Builds profile markdown, evaluation cards, dashboard charts, and system insights.
- Connects voice handlers to UI callbacks:
  - `transcribe_question_handler`
  - `read_answer_handler`

### `app/ui.py`

Defines the Gradio interface and custom CSS. It includes:

- Login/signup screen.
- Main chat workspace.
- Voice recording input through `gr.Audio`.
- Answer audio output through `gr.Audio`.
- Learner dashboard.
- Retrieved examples panel.
- Evaluation panel.
- System insights panel.
- Matplotlib chart outputs.

### `agents/tutor_agent.py`

This file owns the main answer-generation pipeline:

1. Predict difficulty with `predict_level`.
2. Detect the topic with `detect_best_topic`.
3. Retrieve examples with `retrieve_examples`.
4. Retrieve weak-area examples with `retrieve_for_weak_areas`.
5. Build memory and evaluation strategy hints.
6. Select the teaching mode.
7. Build the final tutor prompt.
8. Call the Gemini model through `complete_chat`.
9. Return answer metadata to the UI.

The tutor prompt now asks the LLM to answer in a student-friendly bullet format:

- 3 to 5 short bullets.
- The first bullet gives the direct answer.
- Each bullet is 1 or 2 short sentences.
- A final `Next step:` bullet is added only when useful.
- Bracket citation markers like `[1]`, `[2]`, and `[3]` are explicitly disabled.

Returned tuple:

```python
(level, confidence, topic, examples, weak_examples, answer, teaching_mode)
```

### `agents/evaluator_agent.py`

Generates a follow-up question and evaluates the learner's reply. Evaluation is validated through a Pydantic model:

```python
understanding_level: "good" | "partial" | "poor"
weak_concepts: list[str]
feedback: str
recommended_action: "advance" | "re-explain" | "give easier example" | "give more practice"
```

If the LLM returns invalid JSON, the evaluator safely falls back to a `partial` evaluation.

### `agents/memory_agent.py`

Defines and updates the learner profile. It tracks:

- Sessions
- Total questions asked
- Last predicted level
- Level history
- Topics seen
- Topic counts
- Mastery scores
- Weak areas
- Used explanation styles
- Recommended next topics
- Last evaluation

Mastery is updated with diminishing returns:

```text
good    -> m += 0.20 * (1 - m)
partial -> m -= 0.06
poor    -> m -= 0.18
```

Scores are clamped between `0.0` and `1.0`.

### `ml/classifier.py`

Loads the local DistilBERT classifier from:

```text
models\distilbert_eduagent_v2
```

The classifier predicts:

```text
beginner
intermediate
advanced
```

The classifier is combined with heuristic calibration for edge cases:

- Short definition-style questions are kept at `beginner`.
- Plain comparison questions are promoted to `intermediate`, even if the model predicts `beginner`.
- Advanced comparison questions are marked `advanced` when they include deeper terms such as `derive`, `convergence`, `bias correction`, `loss landscape`, or `mathematically`.
- Explicit advanced-intent terms can override the raw model label.

This prevents cases like `compare Adam optimizer and rms prop` from being incorrectly shown as `advanced` just because the word `optimizer` contains `optimize`.

### `ml/topic_detector.py`

Detects the most relevant topic for the learner question. It uses:

- Alias matching for common topic names and abbreviations.
- Semantic similarity through `sentence-transformers`.
- TF-IDF fallback when embeddings are unavailable.

### `ml/embedder.py`

Loads the shared semantic embedding model:

```python
SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
```

The app now prefers the locally cached MiniLM model to avoid failing on Hugging Face network checks. When the local cache is present, semantic RAG is active with 384-dimensional embeddings. If the cache is missing or loading fails, retrieval falls back to TF-IDF.

### `ml/retriever.py`

Implements the RAG layer.

Pass 1 retrieves examples similar to the user question:

```python
retrieve_examples(user_question, level, top_n=RAG_TOP_N)
```

Pass 2 retrieves examples that target weak concepts:

```python
retrieve_for_weak_areas(
    weak_concepts,
    topic,
    level,
    top_n=RAG_WEAK_TOP_N,
    exclude_questions=already_retrieved,
)
```

Retrieval uses locally cached `all-MiniLM-L6-v2` embeddings when available and TF-IDF otherwise.

### `voice/`

The voice module provides:

- Speech-to-text using Whisper.
- Text-to-speech using a TTS engine wrapper.
- Utility functions used by Gradio callbacks.

Important files:

```text
voice/speech_to_text.py
voice/text_to_speech.py
voice/utils.py
```

The UI uses:

```python
gr.Audio(sources=["microphone"], ...)
```

When the learner stops recording, the audio is transcribed into the normal question textbox. When the learner clicks the read/listen control, the latest answer is synthesized to a `.wav` file and played in the UI.

---

## Setup On Windows PowerShell

From the project root:

```powershell
cd C:\Users\sneha\OneDrive\Desktop\Eduagent
```

Create or recreate the virtual environment:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the existing `.venv` is broken or missing activation files, recreate it:

```powershell
Rename-Item .venv .venv_old
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root. The project already includes `.env.example`.

Minimum required variable:

```text
GEMINI_API_KEY=your_gemini_api_key_here
```

Recommended variables:

```text
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.0-flash
LLM_TIMEOUT_SECONDS=8
LLM_MAX_RETRIES=2
RAG_TOP_N=3
RAG_WEAK_TOP_N=2
CLASSIFIER_HF_REPO=SSneha2005/Eduagent_distilbert
EVAL_MODEL=gemini-2.0-flash
```

Optional database override:

```text
TUTORWEAVE_DB_FILE=runtime\eduagent_app.db
```

The app currently raises an error if `GEMINI_API_KEY` is missing because LLM features are imported at startup.

---

## API Configuration
To run this project, configure a Gemini 2.0 Flash API key in the appropriate environment or configuration file, as the default implementation is set up for this model. The architecture is LLM-independent, so other compatible language models or API providers can be used by making the necessary changes to the model initialization and API integration code. Ensure that the corresponding API credentials and endpoint configurations are updated before execution.

---

## Run The App

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Launch Gradio:

```powershell
python .\gradio_app.py
```

Open:

```text
http://127.0.0.1:7860
```

Because `gradio_app.py` currently uses `share=True`, Gradio may also print a temporary public share URL.

---

## How To Use The App

1. Start the app.
2. Sign up with name, optional username, email, and password.
3. Log in.
4. Ask an AI/ML question in the chat box.
5. Optionally record a voice question; the recording will be transcribed into the question input.
6. Submit the question.
7. Review:
   - Tutor answer
   - Predicted difficulty level
   - Confidence scores
   - Detected topic
   - Retrieved examples
   - System insights
8. Answer the follow-up question in the evaluation section.
9. Check the updated profile dashboard and charts.
10. Use answer audio to listen to the tutor response.

---

## Voice Requirements

Voice input uses `openai-whisper`. Whisper commonly needs `ffmpeg` available on your system PATH.

Install Python voice dependencies through:

```powershell
python -m pip install -r requirements.txt
```

If transcription fails, check:

- `openai-whisper` installed correctly.
- `ffmpeg` is installed and available in PATH.
- The browser has microphone permission.
- The Gradio app is running in a browser tab that can access the microphone.

Text-to-speech uses `pyttsx3` fallback behavior. On Windows, TTS depends on local speech support installed with the OS.

---

## View User Data

The active SQLite database is:

```text
runtime\eduagent_app.db
```

Open it with DB Browser for SQLite, or inspect it with Python.

View users without printing password hashes:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; conn=sqlite3.connect('runtime/eduagent_app.db'); conn.row_factory=sqlite3.Row; rows=conn.execute('SELECT id, username, name, email, created_at FROM users ORDER BY id').fetchall(); [print(dict(r)) for r in rows]; conn.close()"
```

View learner profile summaries:

```powershell
.\.venv\Scripts\python.exe -c "import sqlite3; conn=sqlite3.connect('runtime/eduagent_app.db'); conn.row_factory=sqlite3.Row; rows=conn.execute('SELECT user_id, sessions, questions_asked, last_level, topics_seen, mastery, weak_areas FROM profiles ORDER BY user_id').fetchall(); [print(dict(r)) for r in rows]; conn.close()"
```

Do not expose `password_hash` values in screenshots or reports.

---

## Validation Commands

Quick classifier source check:

```powershell
.\.venv\Scripts\python.exe -c "from config.settings import CLASSIFIER_PATH; import ml.classifier as c; print(CLASSIFIER_PATH); print(c._classifier_source)"
```

Expected source:

```text
local (...\models\distilbert_eduagent_v2)
```

Check classifier calibration:

```powershell
.\.venv\Scripts\python.exe -B -c "from ml.classifier import predict_level; qs=['What is gradient descent?','Compare RMSProp and Adam','compare Adam optimizer and rms prop','Compare Adam and RMSProp in terms of convergence']; [print(q, '->', predict_level(q)[0]) for q in qs]"
```

Expected labels:

```text
What is gradient descent? -> beginner
Compare RMSProp and Adam -> intermediate
compare Adam optimizer and rms prop -> intermediate
Compare Adam and RMSProp in terms of convergence -> advanced
```

Check semantic embedding availability:

```powershell
.\.venv\Scripts\python.exe -B -c "from ml.embedder import semantic_available, embed_model; print(semantic_available); print(embed_model.get_sentence_embedding_dimension() if embed_model else None)"
```

Expected output when semantic RAG is active:

```text
True
384
```

Build/import check:

```powershell
python -c "from app.main import create_app; app = create_app(); print('ok')"
```

Run unit tests with `unittest`:

```powershell
python -m unittest discover tests
```

Run unit tests with `pytest`:

```powershell
python -m pip install pytest
python -m pytest tests -v
```

Run personalization evaluation:

```powershell
python -m eval.run_evaluation
```

Run RAG evaluation:

```powershell
python .\rag_evaluation.py
```

---

## Important Runtime Files

```text
runtime/eduagent_app.db
```

Main SQLite database.

```text
eval/evaluation_results.md
```

Generated personalization evaluation report.

```text
eval/rag_eval_report.md
```

Generated RAG evaluation report.

```text
models/distilbert_eduagent_v2/model.safetensors
```

Primary local DistilBERT classifier weights.

---

## Configuration Reference

Main settings live in [`config/settings.py`](config/settings.py).

| Setting | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required API key for Gemini LLM calls |
| `MODEL_NAME` | Tutor model, default `gemini-2.0-flash` |
| `DATASET_FILE` | Active RAG dataset path |
| `CLASSIFIER_PATH` | Primary local DistilBERT model path |
| `CLASSIFIER_HF_REPO` | Hugging Face fallback classifier repo |
| `DB_FILE` | SQLite database path |
| `LLM_TIMEOUT_SECONDS` | LLM request timeout |
| `LLM_MAX_RETRIES` | Retry count after failed LLM calls |
| `RAG_TOP_N` | Number of normal retrieved examples |
| `RAG_WEAK_TOP_N` | Number of weak-area retrieved examples |
| `EVAL_MODEL` | Evaluation model setting, default `gemini-2.0-flash` |
| `N_EVAL_SAMPLES` | Number of RAG evaluation samples |

---

## RAG Details

TutorWeave uses retrieval-augmented generation rather than sending the learner question directly to the LLM.

The retrieval source is the configured CSV dataset. Each row contains an educational question/answer pair with difficulty and topic metadata.

Semantic retrieval uses the locally cached `sentence-transformers/all-MiniLM-L6-v2` model through `SentenceTransformer(..., local_files_only=True)`. This avoids runtime failures caused by blocked or unavailable Hugging Face network checks. If the local MiniLM cache is not available, the app automatically uses TF-IDF retrieval.

The first retrieval pass uses the learner's question and predicted difficulty level to find relevant examples. The second retrieval pass uses weak concepts stored in the learner profile, such as `"chain rule"` or `"learning rate scheduling"`, to retrieve additional examples that address known gaps.

The tutor prompt includes:

- Learner question
- Predicted level
- Detected topic
- Learner memory hint
- Recent evaluator strategy hint
- Retrieved examples
- Weak-area retrieved examples
- Mode-specific tutoring instructions

The LLM is instructed to use retrieved examples as supporting context without copying them directly.

The tutor is also instructed not to include bracket citations like `[1]`, `[2]`, or `[3]`; retrieved examples remain visible in the UI's retrieved examples panel instead.

---

## Teaching Modes

| Mode | When Used | Behavior |
|---|---|---|
| `default` | No strong personalization signal | Balanced explanation matching predicted level |
| `remedial` | Low mastery, repeated weak areas, or evaluator recommends revisiting | Simpler language, concrete example, re-teaches from scratch |
| `clarification` | Partial understanding or weak concepts detected | Focuses on the confusing concept without repeating everything |
| `advance` | High mastery and repeated topic exposure | Deeper explanation with next-step connections |

---

## Learner Profile Schema

| Field | Type | Meaning |
|---|---|---|
| `sessions` | int | Number of logins/sessions |
| `questions_asked` | int | Total questions asked |
| `last_level` | str | Most recent predicted level |
| `level_history` | list | Sequence of predicted levels |
| `topics_seen` | list | Topics encountered by the learner |
| `topic_counts` | dict | Number of visits/questions per topic |
| `mastery` | dict | Mastery score by topic |
| `weak_areas` | dict | Weak concepts by topic |
| `used_explanations` | dict | Explanation styles already used by topic |
| `recommended_next_topics` | list | Evaluator recommendations |
| `last_evaluation` | dict | Latest structured evaluation |

---

## Common Issues

| Issue | Fix |
|---|---|
| `.venv\Scripts\Activate.ps1` is missing | Recreate the virtual environment: `Rename-Item .venv .venv_old`, then `python -m venv .venv` |
| PowerShell blocks activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then `.\.venv\Scripts\Activate.ps1` |
| Package import error after pulling changes | Activate `.venv` and rerun `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: google.generativeai` | Install updated requirements, or run `python -m pip install google-generativeai` |
| Missing `GEMINI_API_KEY` | Add `GEMINI_API_KEY=...` to `.env`; the app imports LLM features at startup |
| Gemini quota or rate-limit errors | Wait for quota reset, reduce repeated calls, or switch `MODEL_NAME`/API key if allowed |
| Classifier loads from Hugging Face instead of local | Confirm `models/distilbert_eduagent_v2/model.safetensors` exists and `CLASSIFIER_PATH` points to the local folder |
| Semantic RAG falls back to TF-IDF | Confirm the local MiniLM cache works with the semantic embedding availability command in Validation Commands |
| Hugging Face network checks fail for MiniLM | This is usually okay if the model is cached; `ml/embedder.py` uses `local_files_only=True` |
| Whisper transcription fails | Install `ffmpeg`, reinstall `openai-whisper`, and check browser microphone permissions |
| Text-to-speech does not play | Confirm `pyttsx3` is installed and Windows local speech support is available |
| Invalid login credentials | Try the email address first; login supports email, username, or display name, but duplicate display names can be confusing |
| Empty dashboard charts | Normal for new users; ask and evaluate a few questions first |
| No follow-up context | Ask a main question before answering the evaluator |
| Gradio app opens old UI | Stop the server, restart `python .\gradio_app.py`, and hard refresh the browser |
| Dependency conflicts in global Python | Use the project `.venv`; avoid running the app with global Python |

---

## Development Notes

- Keep `.env` private.
- Do not commit real API keys.
- Avoid printing password hashes.
- Use the local virtual environment for all commands.
- The local model weights are large; Git operations may be slower.
- `runtime/` files are generated app state.
- The app is suitable for demos and academic evaluation, not production multi-user deployment without additional hardening.

---

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Gradio |
| LLM | Gemini API through `google-generativeai` |
| Tutor model | `gemini-2.0-flash` by default |
| Difficulty classifier | Fine-tuned DistilBERT via Hugging Face Transformers |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` |
| Retrieval fallback | scikit-learn TF-IDF |
| Database | SQLite |
| Auth | passlib PBKDF2-SHA256 |
| Evaluation schema | Pydantic |
| Charts | Matplotlib |
| Voice input | openai-whisper |
| Voice output | pyttsx3 / local TTS wrapper |
| Data | pandas, NumPy, scikit-learn |

---

## Summary

TutorWeave demonstrates a complete adaptive tutoring loop:

```text
Login
-> Ask question by text or voice
-> DistilBERT level classification
-> Topic detection
-> Two-pass personalized RAG
-> Memory hint generation
-> Teaching mode selection
-> LLM tutor answer
-> Follow-up question
-> Learner reply
-> Pydantic-validated evaluation
-> Mastery and weak-area update
-> SQLite profile save
-> Dashboard refresh
-> Optional answer audio playback
```

The central idea is personalization: two learners can ask similar questions and receive different explanation styles because TutorWeave uses stored mastery, weak areas, prior explanation styles, and evaluation history to guide the next answer.
