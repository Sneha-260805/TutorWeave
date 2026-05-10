"""
EduAgent Dataset Easy Batch Generator v2 Safe
--------------------------------------------
Smaller, safer batch generation to avoid malformed large JSON responses.

Design:
1. Generate multiple candidate Q/A rows per cell in one Gemini JSON call.
2. Apply deterministic local quality filters: length, formatting, duplicate checks, level-label leakage.
3. Save accepted rows continuously to JSONL + CSV.
4. No Gemini self-validator in the hot loop by default, because it can reject usable rows and waste calls.

.env example for test:
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
TARGET_PER_CELL=2
RUN_LIMIT_ROWS=30
OUT_DIR=data_easy_v2_test
MAX_CANDIDATES_PER_CALL=4

.env example for full dataset:
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
TARGET_PER_CELL=8
RUN_LIMIT_ROWS=0
OUT_DIR=data_easy_v2_final
MAX_CANDIDATES_PER_CALL=4
"""

import os
import re
import json
import time
import random
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from difflib import SequenceMatcher

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# =========================================================
# CONFIG
# =========================================================
load_dotenv()


def getenv_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}. Using default {default}.")
        return default


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing. Add it to your .env file.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
TARGET_PER_CELL = getenv_int("TARGET_PER_CELL", 8)
RUN_LIMIT_ROWS = getenv_int("RUN_LIMIT_ROWS", 0)  # 0 = no limit
OUT_DIR = Path(os.getenv("OUT_DIR", "data_easy"))
BATCH_EXTRA_CANDIDATES = getenv_int("BATCH_EXTRA_CANDIDATES", 2)
MAX_CANDIDATES_PER_CALL = getenv_int("MAX_CANDIDATES_PER_CALL", 4)
MAX_BATCH_ATTEMPTS_PER_CELL = getenv_int("MAX_BATCH_ATTEMPTS_PER_CELL", 8)
SLEEP_SECONDS = float(os.getenv("SLEEP_SECONDS", "0.35"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
ACCEPTED_JSONL = OUT_DIR / "accepted_rows.jsonl"
REJECTED_JSONL = OUT_DIR / "rejected_rows.jsonl"
CSV_FILE = OUT_DIR / "eduagent_dataset_easy_v2.csv"
PROGRESS_FILE = OUT_DIR / "progress.json"
PROMPT_VERSION = "eduagent_easy_batch_v2_safe_small_json"

client = genai.Client(api_key=GOOGLE_API_KEY)


# =========================================================
# TOPICS + SUBTOPICS
# =========================================================
TOPIC_TAXONOMY = {
    "machine learning basics": [
        "supervised learning",
        "unsupervised learning",
        "features and labels",
        "training and testing",
    ],
    "neural networks": [
        "neurons and layers",
        "activation functions",
        "forward propagation",
        "weights and biases",
    ],
    "gradient descent": [
        "loss function",
        "learning rate",
        "parameter update",
        "convergence",
    ],
    "backpropagation": [
        "chain rule",
        "gradient flow",
        "weight update",
        "vanishing gradients",
    ],
    "overfitting and regularization": [
        "overfitting",
        "underfitting",
        "dropout",
        "L1 and L2 regularization",
    ],
    "transformer models": [
        "self attention",
        "positional encoding",
        "encoder decoder structure",
        "multi head attention",
    ],
    "attention mechanism": [
        "query key value",
        "attention scores",
        "scaled dot product attention",
        "context vectors",
    ],
    "reinforcement learning": [
        "agent environment interaction",
        "reward signal",
        "policy",
        "exploration and exploitation",
    ],
    "decision trees": [
        "splitting criteria",
        "entropy and information gain",
        "tree depth",
        "pruning",
    ],
    "clustering algorithms": [
        "k means clustering",
        "hierarchical clustering",
        "distance metrics",
        "cluster evaluation",
    ],
    "classification": [
        "binary classification",
        "multiclass classification",
        "decision boundary",
        "class imbalance",
    ],
    "natural language processing": [
        "tokenization",
        "word embeddings",
        "sequence modeling",
        "text classification",
    ],
    "computer vision": [
        "image classification",
        "object detection",
        "feature extraction",
        "image preprocessing",
    ],
    "convolutional neural networks": [
        "convolution filters",
        "pooling",
        "feature maps",
        "CNN architecture",
    ],
    "agentic AI": [
        "planning",
        "tool use",
        "memory",
        "agent feedback loop",
    ],
    "generative adversarial networks": [
        "generator",
        "discriminator",
        "adversarial training",
        "mode collapse",
    ],
    "large language models": [
        "pretraining",
        "fine tuning",
        "context window",
        "instruction following",
    ],
    "vector databases": [
        "embeddings",
        "similarity search",
        "indexing",
        "nearest neighbor retrieval",
    ],
    "prompt engineering": [
        "zero shot prompting",
        "few shot prompting",
        "chain of thought prompting",
        "prompt constraints",
    ],
    "transfer learning": [
        "feature reuse",
        "fine tuning",
        "domain adaptation",
        "frozen layers",
    ],
    "model evaluation metrics": [
        "accuracy",
        "precision and recall",
        "F1 score",
        "confusion matrix",
    ],
    "ethical AI and bias": [
        "dataset bias",
        "fairness",
        "explainability",
        "responsible AI",
    ],
    "retrieval augmented generation": [
        "retrieval",
        "chunking",
        "grounding",
        "retriever generator pipeline",
    ],
    "recurrent neural networks": [
        "hidden state",
        "sequence memory",
        "LSTM",
        "GRU",
    ],
    "graph neural networks": [
        "nodes and edges",
        "message passing",
        "graph embeddings",
        "node classification",
    ],
}

LEVELS = ["beginner", "intermediate", "advanced"]

ANGLE_BANK = {
    "beginner": [
        "definition with simple example",
        "real-world analogy",
        "why it matters",
        "common misconception",
        "basic comparison",
        "high-level working",
        "first learning step",
        "simple consequence",
    ],
    "intermediate": [
        "implementation detail",
        "trade-off analysis",
        "debugging scenario",
        "evaluation decision",
        "dataset impact",
        "workflow choice",
        "comparison between methods",
        "practical failure mode",
    ],
    "advanced": [
        "edge case analysis",
        "scalability concern",
        "theoretical intuition",
        "research limitation",
        "robustness concern",
        "system design integration",
        "probabilistic interpretation",
        "complexity analysis",
    ],
}


# =========================================================
# SCHEMA FOR BATCH GENERATION
# =========================================================
class GeneratedQA(BaseModel):
    question: str = Field(description="One clear AI/ML tutoring question ending with a question mark")
    answer: str = Field(description="Plain-prose answer appropriate to the requested difficulty")
    question_angle: str = Field(description="The reasoning angle used for this question")
    key_concepts: List[str] = Field(description="Two to five key concepts covered")
    expected_reasoning: str = Field(description="What the learner should understand after reading the answer")
    possible_learner_confusion: str = Field(description="A likely confusion this row helps address")
    difficulty_rationale: str = Field(description="Why the row fits the requested level")
    why_this_fits_level: str = Field(description="Short explanation of level fit")


class BatchResponse(BaseModel):
    rows: List[GeneratedQA]


# =========================================================
# HELPERS
# =========================================================
def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(*parts: str) -> str:
    return hashlib.md5("::".join(parts).encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[WARN] Skipped bad JSONL line in {path}")
    return rows


def stringify_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def save_csv(rows: Optional[List[Dict[str, Any]]] = None) -> None:
    rows = rows if rows is not None else load_jsonl(ACCEPTED_JSONL)
    if not rows:
        return
    clean_rows = [{k: stringify_for_csv(v) for k, v in r.items()} for r in rows]
    pd.DataFrame(clean_rows).drop_duplicates(subset=["id"]).to_csv(CSV_FILE, index=False, encoding="utf-8")


def save_progress(total_accepted: int, total_this_run: int) -> None:
    payload = {
        "timestamp": now_utc(),
        "total_accepted": total_accepted,
        "total_this_run": total_this_run,
        "target_per_cell": TARGET_PER_CELL,
        "run_limit_rows": RUN_LIMIT_ROWS,
        "gemini_model": GEMINI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "max_candidates_per_call": MAX_CANDIDATES_PER_CALL,
        "csv_file": str(CSV_FILE),
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(str(text).strip().split())


def clean_question(text: str) -> str:
    q = str(text or "").strip()
    q = q.replace("```", "").strip()
    q = re.sub(r"^question\s*:\s*", "", q, flags=re.I).strip()
    q = q.strip('"').strip("'").strip()
    q = re.sub(r"^[-*\d.)\s]+", "", q).strip()
    q = re.sub(r"\s+", " ", q)
    if not q.endswith("?"):
        q = q.rstrip(".!;:") + "?"
    return q


def clean_answer(text: str) -> str:
    a = str(text or "").strip()
    a = a.replace("```", "").strip()
    a = re.sub(r"^answer\s*:\s*", "", a, flags=re.I).strip()
    a = a.strip('"').strip("'").strip()
    a = re.sub(r"\s*[-*]\s+", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a


def clean_json_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    start_obj = raw.find("{")
    start_arr = raw.find("[")

    if start_obj == -1 and start_arr == -1:
        raise ValueError(f"No JSON found in response: {raw[:300]}")

    if start_arr != -1 and (start_obj == -1 or start_arr < start_obj):
        start = start_arr
        end = raw.rfind("]")
        data = json.loads(raw[start:end + 1])
        return {"rows": data}

    start = start_obj
    end = raw.rfind("}")
    return json.loads(raw[start:end + 1])


BAD_LEVEL_PHRASES = [
    "for a beginner",
    "for an intermediate",
    "for an advanced",
    "as a beginner",
    "beginner:",
    "intermediate:",
    "advanced:",
    "difficulty level",
]

QUESTION_LIMITS = {
    "beginner": (7, 45),
    "intermediate": (9, 60),
    "advanced": (11, 75),
}

ANSWER_LIMITS = {
    "beginner": (50, 130),
    "intermediate": (75, 170),
    "advanced": (95, 230),
}


def has_level_leakage(text: str) -> bool:
    lowered = str(text).lower()
    return any(phrase in lowered for phrase in BAD_LEVEL_PHRASES)


def is_near_duplicate(question: str, existing_questions: List[str]) -> bool:
    q_norm = normalize_text(question)
    if not q_norm:
        return True

    q_tokens = set(q_norm.split())
    for existing in existing_questions:
        e_norm = normalize_text(existing)
        if not e_norm:
            continue
        if q_norm == e_norm:
            return True
        ratio = SequenceMatcher(None, q_norm, e_norm).ratio()
        e_tokens = set(e_norm.split())
        jaccard = len(q_tokens & e_tokens) / max(1, len(q_tokens | e_tokens))
        if ratio >= 0.90 or jaccard >= 0.82:
            return True
    return False


def local_quality_check(row: Dict[str, Any], level: str, existing_questions: List[str]) -> Tuple[bool, str]:
    q = clean_question(row.get("question", ""))
    a = clean_answer(row.get("answer", ""))

    if not q or not a:
        return False, "empty_question_or_answer"

    if has_level_leakage(q) or has_level_leakage(a):
        return False, "level_label_leakage"

    q_words = word_count(q)
    a_words = word_count(a)
    q_min, q_max = QUESTION_LIMITS[level]
    a_min, a_max = ANSWER_LIMITS[level]

    if q_words < q_min:
        return False, f"question_too_short_{q_words}_min_{q_min}"
    if q_words > q_max:
        return False, f"question_too_long_{q_words}_max_{q_max}"
    if a_words < a_min:
        return False, f"answer_too_short_{a_words}_min_{a_min}"
    if a_words > a_max:
        return False, f"answer_too_long_{a_words}_max_{a_max}"

    if is_near_duplicate(q, existing_questions):
        return False, "duplicate_or_near_duplicate"

    # Avoid very generic, low-value questions.
    generic_patterns = [
        r"^what is [a-z0-9\s-]+\?$",
        r"^define [a-z0-9\s-]+\?$",
        r"^explain [a-z0-9\s-]+\?$",
    ]
    if level != "beginner" and any(re.match(p, q.lower()) for p in generic_patterns):
        return False, "too_generic_for_level"

    return True, "ok"


def parsed_to_dict(response: Any) -> Dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        if hasattr(parsed, "dict"):
            return parsed.dict()
        if isinstance(parsed, dict):
            return parsed
    return clean_json_text(getattr(response, "text", ""))


# =========================================================
# PROMPT + MODEL CALL
# =========================================================
def build_batch_prompt(
    topic: str,
    subtopic: str,
    level: str,
    needed: int,
    candidate_count: int,
    existing_questions: List[str],
) -> str:
    angles = ANGLE_BANK[level]
    random.shuffle(angles)
    existing_block = "\n".join(f"- {q}" for q in existing_questions[-12:]) or "None"
    q_min, q_max = QUESTION_LIMITS[level]
    a_min, a_max = ANSWER_LIMITS[level]

    return f"""
You are generating high-quality rows for an AI/ML tutoring dataset called EduAgent.

Generate {candidate_count} diverse candidate rows.
Only generate rows for this exact cell:
Topic: {topic}
Subtopic: {subtopic}
Difficulty level: {level}

Existing questions to avoid:
{existing_block}

Use these angles across the rows where possible:
{', '.join(angles)}

Hard quality rules:
- Each question must be specific, useful, and end with a question mark.
- Question length must be {q_min} to {q_max} words.
- Answer length must be {a_min} to {a_max} words.
- Do not mention the difficulty label in the question or answer.
- Do not write "For a beginner", "For an intermediate", or "For an advanced".
- Do not use markdown, bullets, numbered lists, tables, or code blocks.
- Avoid generic questions like "What is {subtopic}?" unless the beginner row adds context and learning value.
- Each answer must be factually correct, self-contained, and pedagogically useful.
- Questions must be meaningfully different from each other and from the existing questions.

Level expectations:
Beginner: simple explanation, intuitive example, no equations, minimal jargon.
Intermediate: practical reasoning, comparison, debugging, implementation, or evaluation.
Advanced: edge cases, assumptions, scalability, theory, robustness, or system design reasoning.

Return valid JSON only in this exact shape:
{{
  "rows": [
    {{
      "question": "...",
      "answer": "...",
      "question_angle": "...",
      "key_concepts": ["...", "..."],
      "expected_reasoning": "...",
      "possible_learner_confusion": "...",
      "difficulty_rationale": "...",
      "why_this_fits_level": "..."
    }}
  ]
}}
""".strip()


def call_gemini_batch(prompt: str, retries: int = 4) -> List[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.45,
                    max_output_tokens=6144,
                    response_mime_type="application/json",
                    response_schema=BatchResponse,
                ),
            )
            data = parsed_to_dict(response)
            rows = data.get("rows", [])
            if not isinstance(rows, list) or not rows:
                raise ValueError("Gemini returned no rows.")
            return rows
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Batch generation failed attempt={attempt}: {str(exc)[:180]}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Batch generation failed after retries. Last error: {last_error}")


def build_final_row(
    raw: Dict[str, Any],
    topic: str,
    subtopic: str,
    level: str,
) -> Dict[str, Any]:
    question = clean_question(raw.get("question", ""))
    answer = clean_answer(raw.get("answer", ""))
    key_concepts = raw.get("key_concepts")
    if not isinstance(key_concepts, list) or not key_concepts:
        key_concepts = [topic, subtopic]

    row = {
        "id": stable_id(topic, subtopic, level, question),
        "topic": topic,
        "subtopic": subtopic,
        "level": level,
        "question_angle": str(raw.get("question_angle", "conceptual reasoning")).strip(),
        "question": question,
        "answer": answer,
        "question_word_count": word_count(question),
        "answer_word_count": word_count(answer),
        "key_concepts": key_concepts,
        "expected_reasoning": str(raw.get("expected_reasoning", f"Learner should understand {subtopic} within {topic}.")).strip(),
        "difficulty_rationale": str(raw.get("difficulty_rationale", f"Fits {level} level for {subtopic}.")).strip(),
        "why_this_fits_level": str(raw.get("why_this_fits_level", f"The row follows {level}-level constraints.")).strip(),
        "concepts_covered": [topic, subtopic],
        "possible_learner_confusion": str(raw.get("possible_learner_confusion", "")).strip(),
        "generator_provider": "gemini",
        "generator_model": GEMINI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "timestamp": now_utc(),
    }
    return row


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    print("\nEduAgent Easy Batch Dataset Generator v2 Safe")
    print("--------------------------------------------")
    print(f"Gemini model:          {GEMINI_MODEL}")
    print(f"Target per cell:       {TARGET_PER_CELL}")
    print(f"Run limit rows:        {RUN_LIMIT_ROWS if RUN_LIMIT_ROWS else 'No limit'}")
    print(f"Output CSV:            {CSV_FILE}")
    print(f"Max candidates/call:   {MAX_CANDIDATES_PER_CALL}")
    print(f"Prompt version:        {PROMPT_VERSION}")
    print("--------------------------------------------\n")

    accepted_rows = load_jsonl(ACCEPTED_JSONL)
    # Deduplicate previously accepted rows on resume.
    unique_by_id = {row.get("id"): row for row in accepted_rows if row.get("id")}
    accepted_rows = list(unique_by_id.values())

    total_this_run = 0
    cells = [(topic, subtopic, level) for topic, subs in TOPIC_TAXONOMY.items() for subtopic in subs for level in LEVELS]
    random.shuffle(cells)

    try:
        for topic, subtopic, level in tqdm(cells, desc="Cells"):
            existing_for_cell = [
                r["question"]
                for r in accepted_rows
                if r.get("topic") == topic and r.get("subtopic") == subtopic and r.get("level") == level
            ]

            already = len(existing_for_cell)
            if already >= TARGET_PER_CELL:
                continue

            batch_attempt = 0
            while already < TARGET_PER_CELL and batch_attempt < MAX_BATCH_ATTEMPTS_PER_CELL:
                if RUN_LIMIT_ROWS and total_this_run >= RUN_LIMIT_ROWS:
                    save_csv(accepted_rows)
                    save_progress(len(accepted_rows), total_this_run)
                    print(f"\nRun limit reached. Saved to {CSV_FILE}")
                    return

                batch_attempt += 1
                missing = TARGET_PER_CELL - already
                # Keep each JSON response small. Large 10–12 row JSON outputs often fail with
                # "Expecting comma delimiter" even when the content is good.
                candidate_count = min(
                    MAX_CANDIDATES_PER_CALL,
                    max(missing + BATCH_EXTRA_CANDIDATES, missing),
                )

                prompt = build_batch_prompt(
                    topic=topic,
                    subtopic=subtopic,
                    level=level,
                    needed=missing,
                    candidate_count=candidate_count,
                    existing_questions=existing_for_cell,
                )

                candidates = call_gemini_batch(prompt)
                accepted_in_batch = 0

                for candidate in candidates:
                    if already >= TARGET_PER_CELL:
                        break
                    if RUN_LIMIT_ROWS and total_this_run >= RUN_LIMIT_ROWS:
                        break

                    row = build_final_row(candidate, topic, subtopic, level)
                    ok, reason = local_quality_check(row, level, existing_for_cell)

                    if ok:
                        append_jsonl(ACCEPTED_JSONL, row)
                        accepted_rows.append(row)
                        existing_for_cell.append(row["question"])
                        already += 1
                        total_this_run += 1
                        accepted_in_batch += 1
                        print(
                            f"[ACCEPT] {topic} | {subtopic} | {level} "
                            f"| q_words={row['question_word_count']} "
                            f"| a_words={row['answer_word_count']} "
                            f"| angle={row['question_angle']}"
                        )
                    else:
                        reject = {
                            "stage": "local_filter",
                            "reason": reason,
                            "topic": topic,
                            "subtopic": subtopic,
                            "level": level,
                            "question": row.get("question"),
                            "answer": row.get("answer"),
                            "question_word_count": row.get("question_word_count"),
                            "answer_word_count": row.get("answer_word_count"),
                            "timestamp": now_utc(),
                        }
                        append_jsonl(REJECTED_JSONL, reject)
                        print(f"[SKIP] {topic} | {subtopic} | {level} | {reason}")

                if total_this_run and total_this_run % 25 == 0:
                    save_csv(accepted_rows)
                    save_progress(len(accepted_rows), total_this_run)

                # If a batch gave zero usable rows, ask again with more candidates next time.
                if accepted_in_batch == 0:
                    time.sleep(1.5)
                else:
                    time.sleep(SLEEP_SECONDS)

            if already < TARGET_PER_CELL:
                print(
                    f"[WARN] Cell underfilled after attempts: {topic} | {subtopic} | {level} "
                    f"accepted={already}/{TARGET_PER_CELL}. Rerun the same script; resume will continue."
                )

        save_csv(accepted_rows)
        save_progress(len(accepted_rows), total_this_run)
        print("\nDone.")
        print(f"Dataset saved to: {CSV_FILE}")
        print(f"Accepted rows log: {ACCEPTED_JSONL}")
        print(f"Rejected rows log: {REJECTED_JSONL}")

    except KeyboardInterrupt:
        save_csv(accepted_rows)
        save_progress(len(accepted_rows), total_this_run)
        print("\nInterrupted. Progress saved.")


if __name__ == "__main__":
    main()
