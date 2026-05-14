"""
RAG Evaluation for EduAgent — Groq LLaMA as LLM Judge
=======================================================
Optimized for minimal token usage and robust failure handling.

Judge model : llama-3.1-8b-instant  (1 combined call per question, not 4)
Default N   : 5 questions
Checkpoint  : eval/rag_eval_checkpoint.json  (auto-saves, --resume to continue)
Modes       : full | retrieval | judge

HOW TO RUN
----------
  python rag_evaluation.py                      # full, 5 questions
  python rag_evaluation.py --mode retrieval     # no LLM, just retrieval metrics
  python rag_evaluation.py --mode judge         # LLM judge only (skip retrieval stats)
  python rag_evaluation.py --n 10              # evaluate 10 questions
  python rag_evaluation.py --resume            # continue from last checkpoint

OUTPUTS
-------
  eval/rag_eval_results.json   — raw per-question data
  eval/rag_eval_results.csv    — spreadsheet-friendly
  eval/rag_eval_report.md      — human-readable markdown
  eval/rag_judge_raw.jsonl     — raw judge outputs for debugging
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

from agents.memory_agent import ensure_profile_structure
from config.settings import EVAL_MODEL, GROQ_API_KEY, N_EVAL_SAMPLES, RAG_TOP_N
from ml.embedder import embed_model, semantic_available
from ml.retriever import retrieve_examples, retrieve_for_weak_areas

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

JUDGE_MODEL = os.getenv("EVAL_MODEL", EVAL_MODEL)
OUTPUT_DIR = ROOT / "eval"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_FILE = OUTPUT_DIR / "rag_eval_checkpoint.json"
RAW_JUDGE_LOG = OUTPUT_DIR / "rag_judge_raw.jsonl"

judge_client = Groq(api_key=GROQ_API_KEY, timeout=25.0)
BLANK_PROFILE = ensure_profile_structure({})


# =============================================================================
# Rate-limit guard
# =============================================================================

class RateLimitGuard:
    """
    Tracks consecutive 429s. After max_consecutive failures, marks quota
    exhausted so the caller skips remaining judge calls gracefully rather
    than hammering the API and multiplying token usage.
    """

    def __init__(self, max_consecutive: int = 3):
        self._max = max_consecutive
        self._consecutive = 0
        self.exhausted = False
        self.total_skipped = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_429(self) -> None:
        self._consecutive += 1
        logger.warning("429 rate-limit count: %d / %d", self._consecutive, self._max)
        if self._consecutive >= self._max:
            self.exhausted = True
            logger.error(
                "Quota appears exhausted after %d consecutive 429s. "
                "Remaining judge calls will be skipped.",
                self._consecutive,
            )

    def should_skip(self) -> bool:
        if self.exhausted:
            self.total_skipped += 1
            return True
        return False


_guard = RateLimitGuard()


def _is_429(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate_limit" in s or "rate limit" in s


# =============================================================================
# JSON extraction — multiple fallback strategies
# =============================================================================

def _extract_json(text: str) -> dict | None:
    """
    Try several strategies to extract a JSON dict from LLM output.
    Returns None only when all strategies fail.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text).strip().strip("`").strip()

    # Strategy 1: direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: extract first {...} block (handles trailing prose)
    m = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: field-by-field regex
    result: dict = {}
    for field in ("faithfulness", "relevancy", "context_use", "level_fit"):
        fm = re.search(rf'["\']?{field}["\']?\s*:\s*([1-5])', text, re.IGNORECASE)
        if fm:
            result[field] = int(fm.group(1))

    rm = re.search(r'["\']?reason["\']?\s*:\s*["\']([^"\']{5,300})["\']', text, re.IGNORECASE)
    if rm:
        result["reason"] = rm.group(1)

    if result:
        return result

    return None


def _safe_score(obj: dict, key: str, default: int = 3) -> int:
    """Always return an int in [1,5].  Falls back to `default` (3 = neutral)."""
    val = obj.get(key, default)
    try:
        return max(1, min(5, int(val)))
    except (TypeError, ValueError):
        return default


# =============================================================================
# Judge prompt  (single combined call = 1 API call per question, not 4)
# =============================================================================

JUDGE_SYSTEM = (
    "You are a strict evaluator of an AI tutoring system. "
    "Reply ONLY with a valid JSON object — no markdown, no text outside the JSON."
)


def _make_judge_prompt(question: str, level: str, context_docs: list[dict], answer: str) -> str:
    # Slightly larger windows for the richer v2 dataset
    ctx_lines = []
    for i, d in enumerate(context_docs[:2]):
        ctx_lines.append(
            f"[{i+1}] Q: {d['question'][:150]}  A: {d['answer'][:220]}"
        )
    ctx = "\n".join(ctx_lines) if ctx_lines else "(none)"

    return (
        f"Rate this AI tutor response on 4 dimensions (1=worst, 5=best).\n\n"
        f"QUESTION: {question[:200]}\n"
        f"STUDENT LEVEL: {level}\n"
        f"RETRIEVED CONTEXT (what the tutor was given):\n{ctx}\n"
        f"TUTOR ANSWER: {answer[:550]}\n\n"
        f"Scoring criteria:\n"
        f"- faithfulness: answer only states things supported by the context (not hallucinated)\n"
        f"- relevancy: answer directly and fully addresses the question\n"
        f"- context_use: tutor explicitly built upon or referenced the retrieved examples\n"
        f"- level_fit: language and depth suit a {level} student\n\n"
        f'Reply ONLY with this JSON (integers 1-5, no other text):\n'
        f'{{"faithfulness": <1-5>, "relevancy": <1-5>, "context_use": <1-5>, '
        f'"level_fit": <1-5>, "reason": "<one sentence>"}}'
    )


# =============================================================================
# Groq judge caller
# =============================================================================

def _log_raw(question: str, raw: str) -> None:
    try:
        with open(RAW_JUDGE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"q": question[:80], "raw": raw}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def call_groq_judge(
    question: str,
    level: str,
    context_docs: list[dict],
    answer: str,
    max_retries: int = 2,
) -> dict:
    """
    Call the Groq judge model.

    Returns a dict with keys: faithfulness, relevancy, context_use, level_fit,
    reason, raw, skipped.  All score fields are None when skipped or parse-failed.
    """
    # Quota-exhausted sentinel — scores are None so they're excluded from averages
    _SKIP = dict(faithfulness=None, relevancy=None, context_use=None,
                 level_fit=None, reason="quota_exhausted", raw="", skipped=True,
                 parse_failed=False)

    if _guard.should_skip():
        return _SKIP

    prompt = _make_judge_prompt(question, level, context_docs, answer)
    raw_text = ""

    for attempt in range(max_retries + 1):
        try:
            resp = judge_client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=110,
            )
            raw_text = resp.choices[0].message.content.strip()
            _guard.record_success()

            parsed = _extract_json(raw_text)
            if parsed is not None:
                scores = {
                    "faithfulness": _safe_score(parsed, "faithfulness"),
                    "relevancy":    _safe_score(parsed, "relevancy"),
                    "context_use":  _safe_score(parsed, "context_use"),
                    "level_fit":    _safe_score(parsed, "level_fit"),
                    "reason":       parsed.get("reason", ""),
                    "raw":          raw_text,
                    "skipped":      False,
                    "parse_failed": False,
                }
                logger.info(
                    "  → judge OK  faith=%s  relev=%s  ctx=%s  level=%s",
                    scores["faithfulness"], scores["relevancy"],
                    scores["context_use"], scores["level_fit"],
                )
                return scores
            logger.warning(
                "Attempt %d/%d: JSON extraction failed. Raw: %s",
                attempt + 1, max_retries + 1, raw_text[:160],
            )

        except Exception as exc:
            if _is_429(exc):
                _guard.record_429()
                if _guard.exhausted:
                    return _SKIP
                wait = min(2 ** (attempt + 1) * 3, 90)
                logger.warning("429 rate limit. Backing off %ds.", wait)
                time.sleep(wait)
            else:
                logger.warning(
                    "Judge call failed (attempt %d/%d): %s",
                    attempt + 1, max_retries + 1, exc,
                )
                if attempt < max_retries:
                    time.sleep(2.0 * (attempt + 1))

    _log_raw(question, raw_text)
    logger.error(
        "Judge parse/call failed after %d attempts — assigning neutral score 3 for all metrics.",
        max_retries + 1,
    )
    # Return neutral 3 (not None / not 0) so aggregates don't collapse
    return dict(
        faithfulness=3, relevancy=3, context_use=3, level_fit=3,
        reason="parse_failed_neutral", raw=raw_text,
        skipped=False, parse_failed=True,
    )


# =============================================================================
# Retrieval evaluation  (no LLM, pure math)
# =============================================================================

def eval_retrieval(item: dict) -> dict:
    q = item["question"]
    true_level = item["level"]
    true_topic = item["topic"]

    results    = retrieve_examples(q, true_level, top_n=RAG_TOP_N)
    ret_qs     = results["question"].tolist()
    ret_topics = results["topic"].tolist()
    ret_levels = results["level"].tolist()

    hit  = q in ret_qs
    rank = (ret_qs.index(q) + 1) if hit else None
    mrr  = (1.0 / rank) if rank else 0.0

    topic_prec = (
        sum(t == true_topic for t in ret_topics) / len(ret_topics) if ret_topics else 0.0
    )
    level_prec = (
        sum(lv == true_level for lv in ret_levels) / len(ret_levels) if ret_levels else 0.0
    )

    avg_sim = max_sim = 0.0
    if semantic_available and ret_qs:
        q_vec   = embed_model.encode([q])
        r_vecs  = embed_model.encode(ret_qs)
        sims    = cosine_similarity(q_vec, r_vecs).flatten()
        avg_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))

    return {
        "hit":               hit,
        "rank":              rank,
        "mrr":               mrr,
        "topic_precision":   topic_prec,
        "level_precision":   level_prec,
        "avg_semantic_sim":  avg_sim,
        "max_semantic_sim":  max_sim,
        "retrieved_questions": ret_qs,
        "retrieved_topics":    ret_topics,
        "retrieved_levels":    ret_levels,
    }


# =============================================================================
# Evaluate one question
# =============================================================================

def _sanitize(text: str, max_len: int = 600) -> str:
    """
    Clean text before embedding in an LLM prompt.
    Collapses whitespace, strips smart quotes, removes triple-quotes.
    """
    t = str(text)
    # Normalise quote variants
    t = t.replace('‘', "'").replace('’', "'")   # left/right single
    t = t.replace('“', '"').replace('”', '"')   # left/right double
    t = t.replace('"""', '"').replace('""', '"')
    # Collapse whitespace
    t = re.sub(r'[\t\r\n]+', ' ', t)
    t = re.sub(r' {2,}', ' ', t)
    return t.strip()[:max_len]


def _sanitize_context_docs(docs: list[dict]) -> list[dict]:
    """Sanitize Q and A fields in context doc dicts before passing to the judge."""
    return [
        {"question": _sanitize(d["question"], 150), "answer": _sanitize(d["answer"], 220)}
        for d in docs
    ]


def evaluate_one(item: dict, idx: int, total: int, mode: str) -> dict:
    from agents.tutor_agent import generate_tutor_response

    q = item["question"]
    logger.info("[%d/%d] %s", idx, total, q[:70])

    result: dict = {
        "question":      q,
        "ground_truth":  item["ground_truth_answer"],
        "true_level":    item["level"],
        "true_topic":    item["topic"],
    }

    # ── Retrieval layer ───────────────────────────────────────────────────────
    if mode in ("full", "retrieval"):
        ret = eval_retrieval(item)
        result.update({
            "hit":                ret["hit"],
            "rank":               ret["rank"],
            "mrr":                ret["mrr"],
            "topic_precision":    ret["topic_precision"],
            "level_precision":    ret["level_precision"],
            "avg_semantic_sim":   ret["avg_semantic_sim"],
            "retrieved_questions": ret["retrieved_questions"],
            "retrieved_topics":   ret["retrieved_topics"],
        })
    else:
        result.update(dict(hit=None, rank=None, mrr=None, topic_precision=None,
                           level_precision=None, avg_semantic_sim=None,
                           retrieved_questions=[], retrieved_topics=[]))

    if mode == "retrieval":
        result.update(dict(detected_level=None, detected_topic=None, teaching_mode=None,
                           answer=None, faithfulness=None, relevancy=None,
                           context_use=None, level_fit=None,
                           judge_reason=None, judge_skipped=False))
        return result

    # ── Tutor generation ──────────────────────────────────────────────────────
    logger.info("  → generating tutor response...")
    examples = pd.DataFrame()
    level = item["level"]     # will be overwritten by generate_tutor_response
    topic = item["topic"]     # will be overwritten by generate_tutor_response
    teaching_mode = "unknown"
    answer = ""

    try:
        level, _conf, topic, examples, _weak, answer, teaching_mode = \
            generate_tutor_response(q, dict(BLANK_PROFILE))
        logger.info(
            "  → detected  level=%-12s  topic=%s  mode=%s",
            level, topic, teaching_mode,
        )
    except Exception as exc:
        logger.warning("  → tutor generation FAILED: %s", exc)
        answer = f"[generation failed: {exc}]"

    level_match = "✓" if level == item["level"] else "✗ (expected %s)" % item["level"]
    topic_match = "✓" if topic == item["topic"] else "✗ (expected %s)" % item["topic"]
    logger.info("  → level %s  topic %s", level_match, topic_match)

    result.update({
        "detected_level":  level,
        "detected_topic":  topic,
        "teaching_mode":   teaching_mode,
        "answer":          (answer or "")[:600],
    })

    if mode == "judge" and len(examples) == 0:
        try:
            examples = retrieve_examples(q, level, top_n=RAG_TOP_N)
        except Exception:
            pass

    # Build context docs from what the tutor actually received; sanitize for judge
    raw_context_docs = [
        {"question": str(row.question), "answer": str(row.answer)}
        for row in examples.itertuples(index=False)
    ] if len(examples) > 0 else []
    context_docs = _sanitize_context_docs(raw_context_docs)

    # ── LLM Judge ─────────────────────────────────────────────────────────────
    logger.info("  → calling judge (model=%s)...", JUDGE_MODEL)
    judge = call_groq_judge(
        _sanitize(q, 200), level, context_docs, _sanitize(answer, 550)
    )
    time.sleep(1.2)  # polite rate-limit pause

    if judge.get("parse_failed"):
        logger.warning("  → judge parse failed — neutral score 3 assigned for all metrics")
    elif judge.get("skipped"):
        logger.warning("  → judge skipped (quota exhausted)")

    result.update({
        "faithfulness":  judge.get("faithfulness"),
        "relevancy":     judge.get("relevancy"),
        "context_use":   judge.get("context_use"),
        "level_fit":     judge.get("level_fit"),
        "judge_reason":  judge.get("reason", ""),
        "judge_skipped": judge.get("skipped", False),
        "judge_parse_failed": judge.get("parse_failed", False),
    })
    return result


# =============================================================================
# Dataset builder
# =============================================================================

def build_eval_dataset(n: int) -> list[dict]:
    df = pd.read_csv(ROOT / "datasets" / "data_easy" / "eduagent_dataset_easy_v2.csv")
    per_level = max(1, n // 3)

    samples = (
        df.groupby(["level", "topic"], group_keys=False)
          .apply(lambda g: g.sample(min(2, len(g)), random_state=42))
          .groupby("level", group_keys=False)
          .apply(lambda g: g.sample(min(per_level, len(g)), random_state=42))
          .reset_index(drop=True)
          .head(n)
    )

    eval_set = [
        {
            "question":            row["question"],
            "ground_truth_answer": row["answer"],
            "level":               row["level"],
            "topic":               row["topic"],
            "subtopic":            row["subtopic"] if "subtopic" in row.index else "",
        }
        for _, row in samples.iterrows()
    ]
    logger.info(
        "Eval dataset: %d questions | %d levels | %d topics",
        len(eval_set), samples["level"].nunique(), samples["topic"].nunique(),
    )
    return eval_set


# =============================================================================
# Checkpoint helpers
# =============================================================================

def load_checkpoint() -> list[dict]:
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Loaded checkpoint: %d completed questions.", len(data))
            return data
        except Exception as exc:
            logger.warning("Could not load checkpoint (%s) — starting fresh.", exc)
    return []


def save_checkpoint(results: list[dict]) -> None:
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("Checkpoint save failed: %s", exc)


# =============================================================================
# Report builders
# =============================================================================

def _avg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def build_report(results: list[dict], mode: str) -> str:
    n = len(results)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    skipped = sum(1 for r in results if r.get("judge_skipped"))

    lines = [
        "# EduAgent — RAG Evaluation Report",
        "",
        f"- **Generated:** {now}",
        f"- **Mode:** {mode}",
        f"- **Questions evaluated:** {n}",
        f"- **Judge model:** {JUDGE_MODEL}",
        f"- **RAG top-N:** {RAG_TOP_N}",
        f"- **Judge calls skipped (rate-limit):** {skipped}",
        "",
        "---",
        "## Scorecard",
        "",
    ]

    if mode != "judge":
        lvl_acc = sum(
            r.get("detected_level") == r.get("true_level")
            for r in results
            if r.get("detected_level") and r.get("true_level")
        ) / max(n, 1)
        lines += [
            "### Retrieval Layer",
            "",
            "| Metric | Score | Notes |",
            "|--------|-------|-------|",
            f"| Hit Rate @{RAG_TOP_N} | **{_avg(results,'hit'):.1%}** | Exact source Q in top-{RAG_TOP_N} |",
            f"| MRR | **{_avg(results,'mrr'):.4f}** | Mean Reciprocal Rank |",
            f"| Topic Precision | **{_avg(results,'topic_precision'):.1%}** | Retrieved docs match correct topic |",
            f"| Level Precision | **{_avg(results,'level_precision'):.1%}** | Retrieved docs match correct level |",
            f"| Avg Semantic Sim | **{_avg(results,'avg_semantic_sim'):.4f}** | Cosine sim query vs retrieved Qs |",
            f"| Level Detection Accuracy | **{lvl_acc:.1%}** | Correctly detected beginner/intermediate/advanced |",
            "",
        ]

    if mode != "retrieval":
        faith = _avg(results, "faithfulness")
        relev = _avg(results, "relevancy")
        cutil = _avg(results, "context_use")
        lvlft = _avg(results, "level_fit")
        lines += [
            "### Generation Layer  (LLM-as-Judge, 1–5)",
            "",
            "| Metric | Avg | What it measures |",
            "|--------|-----|-----------------|",
            f"| Faithfulness | **{faith:.2f}** | Answer grounded in context (not hallucinated) |",
            f"| Answer Relevancy | **{relev:.2f}** | Answer addresses the student's question |",
            f"| Context Utilization | **{cutil:.2f}** | Tutor built upon retrieved examples |",
            f"| Level Appropriateness | **{lvlft:.2f}** | Explanation pitched at the right difficulty |",
            "",
        ]

    lines += ["---", "## Per-Question Results", ""]

    for i, r in enumerate(results, 1):
        short_q = r["question"][:80] + ("…" if len(r["question"]) > 80 else "")
        lines.append(f"### Q{i}. {short_q}")
        lines.append(f"- **True:** `{r.get('true_level')}` / `{r.get('true_topic')}`")

        if r.get("detected_level"):
            ok = "✓" if r["detected_level"] == r["true_level"] else "✗ MISMATCH"
            lines.append(
                f"- **Detected:** `{r['detected_level']}` ({ok}) / `{r.get('detected_topic', '')}`"
            )

        if mode != "judge" and r.get("mrr") is not None:
            lines.append(
                f"- **Retrieval:** hit={'YES' if r['hit'] else 'NO'}  "
                f"mrr={r['mrr']:.3f}  sim={r.get('avg_semantic_sim', 0):.3f}  "
                f"topic_prec={r.get('topic_precision', 0):.0%}  level_prec={r.get('level_precision', 0):.0%}"
            )

        if mode != "retrieval":
            if r.get("judge_skipped"):
                lines.append("- **Judge:** *skipped — quota exhausted*")
            elif r.get("judge_parse_failed"):
                lines.append(
                    f"- **Judge:** *parse failed — neutral 3 assigned*  "
                    f"faith=3/5  relev=3/5  ctx=3/5  level=3/5"
                )
            elif r.get("faithfulness") is None:
                lines.append("- **Judge:** *no score (generation failed)*")
            else:
                lines.append(
                    f"- **Judge:** faith={r['faithfulness']}/5  "
                    f"relev={r['relevancy']}/5  "
                    f"ctx={r['context_use']}/5  "
                    f"level={r['level_fit']}/5"
                )
                if r.get("judge_reason"):
                    lines.append(f"  > {r['judge_reason']}")

        lines.append("")

    # Weakness analysis
    missed    = [r for r in results if r.get("hit") is False]
    low_faith = [r for r in results if isinstance(r.get("faithfulness"), int) and r["faithfulness"] <= 2]
    low_ctx   = [r for r in results if isinstance(r.get("context_use"), int) and r["context_use"] <= 2]
    low_lvl   = [r for r in results if isinstance(r.get("level_fit"), int) and r["level_fit"] <= 2]

    lines += ["---", "## Weaknesses & Recommendations", ""]

    if missed:
        lines += [
            f"**Retrieval misses ({len(missed)} questions not found in top-{RAG_TOP_N}):**",
            "Fix: Increase RAG_TOP_N, or improve topic detection.",
        ] + [f"- {r['question'][:90]}" for r in missed] + [""]

    if low_faith:
        lines += [
            f"**Low faithfulness ({len(low_faith)} questions — answer went beyond context):**",
            "Fix: Add 'Only use information from the retrieved examples' to system prompt.",
        ] + [f"- {r['question'][:90]}" for r in low_faith] + [""]

    if low_ctx:
        lines += [
            f"**Low context utilization ({len(low_ctx)} questions):**",
            "Fix: Add 'You MUST directly reference the retrieved examples' to system prompt.",
        ] + [f"- {r['question'][:90]}" for r in low_ctx] + [""]

    if low_lvl:
        lines += [
            f"**Wrong level calibration ({len(low_lvl)} questions):**",
            "Fix: Review DistilBERT classifier training data or improve heuristics.",
        ] + [f"- {r['question'][:90]}" for r in low_lvl] + [""]

    if not any([missed, low_faith, low_ctx, low_lvl]):
        lines.append("No significant weaknesses found across all metrics.")

    return "\n".join(lines)


def build_csv(results: list[dict]) -> str:
    rows = []
    for r in results:
        rows.append({
            "question":         r.get("question", "")[:100],
            "true_level":       r.get("true_level", ""),
            "true_topic":       r.get("true_topic", ""),
            "detected_level":   r.get("detected_level", ""),
            "detected_topic":   r.get("detected_topic", ""),
            "hit":              r.get("hit", ""),
            "mrr":              r.get("mrr", ""),
            "topic_precision":  r.get("topic_precision", ""),
            "level_precision":  r.get("level_precision", ""),
            "avg_semantic_sim": r.get("avg_semantic_sim", ""),
            "faithfulness":     r.get("faithfulness", ""),
            "relevancy":        r.get("relevancy", ""),
            "context_use":      r.get("context_use", ""),
            "level_fit":        r.get("level_fit", ""),
            "judge_skipped":    r.get("judge_skipped", False),
            "judge_reason":     r.get("judge_reason", ""),
        })
    return pd.DataFrame(rows).to_csv(index=False)

def eval_weak_area_retrieval():
    """
    Evaluates Pass 2 — retrieve_for_weak_areas() — independently.
    Simulates learner profiles with known weak concepts and checks
    whether retrieved examples are semantically relevant to those concepts.
    """
    logger.info("\n--- Evaluating Pass 2: Weak Area RAG ---")

    # Simulate realistic weak-area profiles a learner might have
    test_cases = [
        {
            "weak_concepts": ["chain rule", "gradient flow"],
            "topic": "backpropagation",
            "level": "intermediate",
            "expected_keywords": ["gradient", "derivative", "chain", "backward"],
        },
        {
            "weak_concepts": ["vanishing gradient", "long-term dependencies"],
            "topic": "recurrent neural networks",
            "level": "advanced",
            "expected_keywords": ["vanishing", "lstm", "gradient", "memory"],
        },
        {
            "weak_concepts": ["overfitting", "bias variance"],
            "topic": "overfitting and regularization",
            "level": "beginner",
            "expected_keywords": ["overfit", "regulariz", "variance", "bias"],
        },
        {
            "weak_concepts": ["attention scores", "query key value"],
            "topic": "attention mechanism",
            "level": "advanced",
            "expected_keywords": ["attention", "query", "key", "value", "softmax"],
        },
        {
            "weak_concepts": ["pooling", "feature maps"],
            "topic": "convolutional neural networks",
            "level": "beginner",
            "expected_keywords": ["pool", "feature", "filter", "convol"],
        },
    ]

    results = []
    for case in test_cases:
        retrieved = retrieve_for_weak_areas(
            weak_concepts=case["weak_concepts"],
            topic=case["topic"],
            level=case["level"],
            top_n=2,
        )

        if len(retrieved) == 0:
            results.append({
                "weak_concepts": case["weak_concepts"],
                "topic": case["topic"],
                "level": case["level"],
                "retrieved_count": 0,
                "keyword_hit_rate": 0.0,
                "retrieved_questions": [],
            })
            continue

        # Check: do retrieved questions contain expected keywords?
        keyword_hits = []
        for _, row in retrieved.iterrows():
            q_lower = row["question"].lower() + " " + row["answer"].lower()
            hit = any(kw in q_lower for kw in case["expected_keywords"])
            keyword_hits.append(hit)

        # Semantic similarity between weak concept queries and retrieved Qs
        if semantic_available:
            concept_query = " ".join(case["weak_concepts"])
            q_vec = embed_model.encode([concept_query])
            r_vecs = embed_model.encode(retrieved["question"].tolist())
            sims = cosine_similarity(q_vec, r_vecs).flatten()
            avg_sim = float(np.mean(sims))
        else:
            avg_sim = 0.0

        results.append({
            "weak_concepts": case["weak_concepts"],
            "topic": case["topic"],
            "level": case["level"],
            "retrieved_count": len(retrieved),
            "keyword_hit_rate": sum(keyword_hits) / len(keyword_hits),
            "avg_semantic_sim": avg_sim,
            "retrieved_questions": retrieved["question"].tolist(),
        })

        logger.info("  Weak: %s → retrieved %d | keyword hit: %.0f%% | sim: %.3f",
            case["weak_concepts"], len(retrieved),
            sum(keyword_hits)/len(keyword_hits)*100, avg_sim)

    # Save results
    with open(OUTPUT_DIR / "pass2_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    avg_kw  = np.mean([r["keyword_hit_rate"] for r in results])
    avg_sim = np.mean([r.get("avg_semantic_sim", 0) for r in results])
    print(f"\n  PASS 2 — WEAK AREA RAG")
    print(f"  Keyword relevance : {avg_kw:.1%}")
    print(f"  Avg semantic sim  : {avg_sim:.4f}")

    return results
# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="EduAgent RAG Evaluation")
    parser.add_argument(
        "--mode", choices=["full", "retrieval", "judge"], default="full",
        help="full=retrieval+judge (default), retrieval=no LLM, judge=LLM only",
    )
    parser.add_argument(
        "--n", type=int, default=N_EVAL_SAMPLES,
        help=f"Number of questions to evaluate (default: {N_EVAL_SAMPLES})",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoint (skip already-completed questions)",
    )
    args = parser.parse_args()

    # Lazy import so classifier loading logs appear before the header
    from ml.classifier import _classifier_available, _classifier_source

    print(f"\n{'='*60}")
    print(f"  EduAgent RAG Evaluation")
    print(f"  Mode: {args.mode.upper()} | N: {args.n} | Judge: {JUDGE_MODEL}")
    clf_label = _classifier_source if _classifier_available else "heuristics-only (no model)"
    print(f"  Classifier: {clf_label}")
    print(f"{'='*60}\n")

    eval_set = build_eval_dataset(args.n)

    results: list[dict] = []
    if args.resume:
        results = load_checkpoint()
        done_qs = {r["question"] for r in results}
        eval_set = [item for item in eval_set if item["question"] not in done_qs]
        logger.info(
            "Resuming: %d already done, %d remaining.", len(results), len(eval_set)
        )

    total = len(results) + len(eval_set)
    for item in eval_set:
        idx = len(results) + 1
        result = evaluate_one(item, idx, total, args.mode)
        results.append(result)
        save_checkpoint(results)   # incremental save — safe if interrupted

    # Write final outputs
    report   = build_report(results, args.mode)
    csv_data = build_csv(results)

    (OUTPUT_DIR / "rag_eval_report.md").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "rag_eval_results.csv").write_text(csv_data, encoding="utf-8")
    with open(OUTPUT_DIR / "rag_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    n = len(results)
    skipped = sum(1 for r in results if r.get("judge_skipped"))

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")

    if args.mode != "judge":
        print(f"\n  RETRIEVAL")
        print(f"  Hit Rate @{RAG_TOP_N}    : {_avg(results, 'hit'):.1%}")
        print(f"  MRR              : {_avg(results, 'mrr'):.4f}")
        print(f"  Topic Precision  : {_avg(results, 'topic_precision'):.1%}")
        print(f"  Level Precision  : {_avg(results, 'level_precision'):.1%}")
        print(f"  Avg Semantic Sim : {_avg(results, 'avg_semantic_sim'):.4f}")
        lvl_acc = sum(
            r.get("detected_level") == r.get("true_level") for r in results
            if r.get("detected_level") and r.get("true_level")
        ) / max(n, 1)
        print(f"  Level Accuracy   : {lvl_acc:.1%}")

    if args.mode != "retrieval":
        print(f"\n  GENERATION  (LLM-Judge /5)  [{skipped} skipped]")
        print(f"  Faithfulness     : {_avg(results, 'faithfulness'):.2f}")
        print(f"  Answer Relevancy : {_avg(results, 'relevancy'):.2f}")
        print(f"  Context Use      : {_avg(results, 'context_use'):.2f}")
        print(f"  Level Fit        : {_avg(results, 'level_fit'):.2f}")

    print(f"\n  Saved: eval/rag_eval_report.md | eval/rag_eval_results.csv")
    print(f"{'='*60}\n")
    pass2_results = eval_weak_area_retrieval()

if __name__ == "__main__":
    main()
