#!/usr/bin/env python3
"""**LLM-as-Judge** — Re-evaluate predicted answers using an LLM judge.

Reads a ``details.jsonl`` from any eval run and uses an LLM to judge whether
each predicted answer is correct, given the full reasoning context (chain
description, evidence plan with modality rationales, and retrieved chunks).

This is more nuanced than exact/alias string matching — the judge can accept
semantically equivalent answers and penalize answers that happen to string-match
but are factually wrong.

Usage::

    python evaluation/eval_judge.py
    python evaluation/eval_judge.py --input /path/to/details.jsonl
    python evaluation/eval_judge.py --max-items 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

from loguru import logger

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from _common import load_config, _normalize, _apply_extra_body, call_llm_with_retry, parse_json_robust


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer."""
    return _normalize(text).split()


def compute_text_metrics(predicted: str, expected: str, aliases: list[str]) -> dict:
    """Compute EM, F1, and BLEU-1 between predicted and expected (best of aliases)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(expected)
    ref_list = [ref_tokens] + [_tokenize(a) for a in (aliases or [])]

    # Exact Match (best of expected + aliases)
    em = int(any(
        _normalize(predicted) == _normalize(ref)
        for ref in [expected] + (aliases or [])
    ))

    # F1 on tokens (best of references)
    best_f1 = 0.0
    for ref in ref_list:
        if not pred_tokens or not ref:
            f1 = 1.0 if pred_tokens == ref else 0.0
        else:
            common = set(pred_tokens) & set(ref)
            if not common:
                f1 = 0.0
            else:
                # Weighted by count
                pred_counts = {t: pred_tokens.count(t) for t in common}
                ref_counts = {t: ref.count(t) for t in common}
                tp = sum(min(pred_counts[t], ref_counts[t]) for t in common)
                precision = tp / len(pred_tokens)
                recall = tp / len(ref)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        best_f1 = max(best_f1, f1)

    # BLEU-1 (best of references)
    smoother = SmoothingFunction().method1
    best_bleu = 0.0
    for ref in ref_list:
        if not ref or not pred_tokens:
            bleu = 1.0 if pred_tokens == ref else 0.0
        else:
            try:
                bleu = sentence_bleu([ref], pred_tokens, weights=(1.0,), smoothing_function=smoother)
            except Exception:
                bleu = 0.0
        best_bleu = max(best_bleu, bleu)

    return {
        "em": em,
        "f1": round(best_f1, 4),
        "bleu1": round(best_bleu, 4),
    }


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------


SCORE_BANDS = """Score bands (how well does the predicted answer match the expected answer):

1.0 — PERFECT. Semantically identical to the expected answer or an alias.
     Only differs in formatting, capitalization, or obvious paraphrase.
     Example: expected="8:15 AM", predicted="8:15 in the morning" → 1.0
     Example: expected="mental health", predicted="mental health awareness" → 1.0

0.7 — MOSTLY CORRECT. Right answer present but with extra context, minor
     redundancy, or slightly imprecise wording that doesn't change meaning.
     Example: expected="a snowboard", predicted="A snowboard with multicolored
     graphic design" → 0.7
     Example: expected="from $12 to $85", predicted="from $12.00 to $85.00" → 1.0

0.5 — PARTIAL. Gets the general category or one part of a multi-part answer
     right, but misses specifics or a sub-part.
     Example: expected="from $12 to $85", predicted="$85" → 0.5
     Example: expected="from pale yellow to soft sage green",
              predicted="sage green" → 0.5

0.3 — ATTEMPTED. Shows an attempt to answer but the core fact is wrong or
     mostly incorrect. May have a faint connection to the question.
     Example: expected="8:15 AM", predicted="around noon" → 0.3

0.0 — WRONG. Completely incorrect, irrelevant, refused to answer
     ("I don't know" on an answerable question), empty, or error.
     Example: expected="8:15 AM", predicted="I don't know" → 0.0"""


def build_judge_prompt(entry: dict) -> dict:
    """Build a minimal judge prompt: question + expected + predicted only.

    Returns a dict with ``messages`` (OpenAI format) and ``ground_truth`` info.
    """
    question = entry.get("question", "")
    expected = entry.get("expected_answer") or entry.get("ground_truth_answer", "")
    aliases = entry.get("answer_aliases", [])
    predicted = entry.get("predicted_answer", "")
    answerability = entry.get("answerability", "answerable")

    system_prompt = (
        "You are an expert evaluator for a memory benchmark. Your task is to "
        "score how well a predicted answer matches the expected answer.\n\n"
        f"{SCORE_BANDS}\n\n"
        "Important:\n"
        "- Judge semantic equivalence, not verbatim string match.\n"
        "- Accepted aliases count as correct answers.\n"
        '- For unanswerable questions, "I don\'t know" or similar refusal '
        "should score 1.0; any other answer should score 0.0.\n"
        "- If the predicted answer is empty, error, or nonsensical → 0.0.\n\n"
        "Respond with a JSON object:\n"
        '{"reasoning": "brief explanation of why this score", "score": <float>}'
    )

    aliases_str = ", ".join(f'"{a}"' for a in aliases) if aliases else "(none)"

    user_prompt = (
        f"Question: {question}\n"
        f"Answerability: {answerability}\n\n"
        f"Expected answer: {expected}\n"
        f"Accepted aliases: {aliases_str}\n\n"
        f"Predicted answer: {predicted}\n\n"
        "Score this answer."
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "expected": expected,
        "aliases": aliases,
        "predicted": predicted,
        "answerability": answerability,
    }


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------


def call_judge(messages: list[dict], judge_config: dict) -> dict:
    """Call the LLM judge and parse the score + reasoning."""
    from openai import OpenAI

    client = OpenAI(timeout=10, 
        api_key=judge_config.get("api_key", "sk-placeholder"),
        base_url=judge_config.get("base_url"),
    )

    kwargs = {
        "model": judge_config.get("llm_model", "gpt-4o"),
        "messages": messages,
        "temperature": judge_config.get("temperature", 0.0),
        "max_tokens": judge_config.get("max_tokens", 256),
    }

    _apply_extra_body(kwargs, judge_config)

    raw = ""
    try:
        raw = call_llm_with_retry(client, kwargs)
        result = parse_json_robust(raw)
        score = float(result.get("score", 0))
        # Clamp to [0, 1] — missing score defaults to 0
        if score < 0 or score > 1:
            logger.debug(
                f"[Judge] score={score} out of range — clamping to [0,1]"
            )
            score = max(0.0, min(1.0, score))
        logger.debug(
            f"[Judge] score={score} reasoning={result.get('reasoning', '')[:100]}"
        )
        result["score"] = score
        result["raw_response"] = raw
        return result
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(
            f"Judge returned invalid response: {e}\n"
            f"  Raw response: {raw[:500]}"
        )
        return {"score": 0, "reasoning": str(e), "raw_response": raw, "error": True}
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return {"score": 0, "reasoning": str(e), "raw_response": raw, "error": True}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_judge_results(results: list[dict]) -> dict:
    """Compute mean judge score and band distribution."""
    total = len(results)
    if total == 0:
        return {"total": 0}

    valid = [r for r in results if not r.get("judge_error")]
    errors = total - len(valid)
    scores = [r["judge_score"] for r in valid]

    # Band distribution
    band_counts: dict[str, int] = defaultdict(int)
    for s in scores:
        if s >= 1.0:
            band_counts["1.0"] += 1
        elif s >= 0.7:
            band_counts["0.7"] += 1
        elif s >= 0.5:
            band_counts["0.5"] += 1
        elif s >= 0.3:
            band_counts["0.3"] += 1
        else:
            band_counts["0.0"] += 1

    def _mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    # Text metrics
    em_scores = [r.get("em", 0) for r in results]
    f1_scores = [r.get("f1", 0.0) for r in results]
    bleu_scores = [r.get("bleu1", 0.0) for r in results]

    # By question type
    by_type: dict = defaultdict(lambda: {"total": 0, "scores": [], "em": [], "f1": [], "bleu1": []})
    for r in results:
        qt = r.get("question_type", "unknown")
        by_type[qt]["total"] += 1
        if not r.get("judge_error"):
            by_type[qt]["scores"].append(r["judge_score"])
        by_type[qt]["em"].append(r.get("em", 0))
        by_type[qt]["f1"].append(r.get("f1", 0.0))
        by_type[qt]["bleu1"].append(r.get("bleu1", 0.0))

    per_type = {}
    for qt, counts in sorted(by_type.items()):
        t = counts["total"]
        per_type[qt] = {
            "count": t,
            "judge_score": round(_mean(counts["scores"]), 4),
            "em": round(_mean(counts["em"]), 4) if t else 0,
            "f1": round(_mean(counts["f1"]), 4) if t else 0,
            "bleu1": round(_mean(counts["bleu1"]), 4) if t else 0,
        }

    return {
        "total": total,
        "errors": errors,
        "judge_score": round(_mean(scores), 4),
        "band_distribution": dict(band_counts),
        "em": _mean(em_scores),
        "f1": _mean(f1_scores),
        "bleu1": _mean(bleu_scores),
        "per_question_type": per_type,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-Judge: Re-evaluate predicted answers"
    )
    p.add_argument(
        "--input", default=None,
        help="Path to details.jsonl from an eval run (default from config)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output dir (default from config, or <input_dir>/<model>_judged)",
    )
    p.add_argument(
        "--config", default=os.path.join(HERE, "config_judge.json"),
        help="Path to judge config JSON",
    )
    p.add_argument(
        "--max-items", type=int, default=None,
        help="Limit number of items to judge",
    )
    p.add_argument(
        "--judge-workers", type=int, default=None,
        help="Parallel judge calls (default: config judge_workers, else 20)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    input_path = args.input
    config = load_config(args.config)
    judge_config = config.get("judge", {})
    judge_workers = args.judge_workers or config.get("judge_workers", 20)

    # Input: CLI arg > config > default
    input_path = args.input or config.get("input")
    if not input_path or not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Output: CLI arg > config > default (<input_dir>/<model>_judged)
    if args.output_dir:
        output_dir = args.output_dir
    elif config.get("output_dir"):
        output_dir = config["output_dir"]
    else:
        base = os.path.dirname(input_path)
        model_slug = judge_config.get("llm_model", "judge").replace("/", "_")
        output_dir = os.path.join(base, f"{model_slug}_judged")

    # Logging
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<level>{level: <8}</level> | {message}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.add(
        os.path.join(output_dir, "judge.log"), level="DEBUG",
        format="{time:HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    # Load entries — supports both .jsonl (details) and .json (report)
    entries = []
    if input_path.endswith(".jsonl"):
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        # Assume .json — supports oracle (items array) and MemoryOS (personas[].items)
        with open(input_path) as f:
            data = json.load(f)

        # MemoryOS format: nested personas → items
        if "personas" in data:
            raw_items = []
            for persona in data["personas"]:
                raw_items.extend(persona.get("items", []))
        else:
            raw_items = data.get("items", data if isinstance(data, list) else [])

        for item in raw_items:
            # Normalize MemoryOS field names
            if "ground_truth_answer" in item and "expected_answer" not in item:
                item["expected_answer"] = item["ground_truth_answer"]
            # Normalize refusal answers to canonical form
            pred = item.get("predicted_answer", "")
            if isinstance(pred, str) and pred.strip().upper() == "UNANSWERABLE":
                item["predicted_answer"] = "I don't know."
            # Normalize missing fields
            if "answerability" not in item:
                item["answerability"] = (
                    "unanswerable" if item.get("question_type") == "answer_refusal"
                    else "answerable"
                )
            if "answer_aliases" not in item:
                item["answer_aliases"] = []
            entries.append(item)

    logger.info(f"Loaded {len(entries)} entries from {input_path}")
    logger.info(f"Judge workers: {judge_workers}")

    if args.max_items and args.max_items < len(entries):
        entries = entries[:args.max_items]
        logger.info(f"Limited to {args.max_items} items")

    # Judge each entry
    results = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    def judge_one(entry):
        prompt = build_judge_prompt(entry)
        judge_result = call_judge(prompt["messages"], judge_config)
        entry["judge_score"] = judge_result.get("score", 0)
        entry["judge_error"] = judge_result.get("error", False)
        entry["judge_reasoning"] = judge_result.get("reasoning", "")
        entry["judge_raw"] = judge_result.get("raw_response", "")
        # Text metrics
        tm = compute_text_metrics(
            entry.get("predicted_answer", ""),
            entry.get("expected_answer") or entry.get("ground_truth_answer", ""),
            entry.get("answer_aliases", []),
        )
        entry["em"] = tm["em"]
        entry["f1"] = tm["f1"]
        entry["bleu1"] = tm["bleu1"]
        return entry

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        futures = {executor.submit(judge_one, e): i for i, e in enumerate(entries)}
        pbar = tqdm(as_completed(futures), total=len(entries), desc="  Judging", unit="item")
        for future in pbar:
            entry = future.result()
            results.append(entry)

    # Sort by original order
    results.sort(key=lambda e: entries.index(e) if e in entries else 999999)

    elapsed = time.time() - t_start
    metrics = aggregate_judge_results(results)

    # Compare with string-match
    string_correct = sum(
        1 for r in results
        if r.get("evaluation", {}).get("alias_match")
        or r.get("evaluation", {}).get("unanswerable_correct")
        or r.get("match")  # oracle report.json uses 'match' boolean
    )
    string_accuracy = round(string_correct / len(results), 4) if results else 0

    summary = {
        "input": input_path,
        "output_dir": output_dir,
        "judged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": judge_config.get("llm_model", "?"),
        "num_items": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "string_match_accuracy": string_accuracy,
        "judge_score": metrics["judge_score"],
        "band_distribution": metrics["band_distribution"],
        "errors": metrics["errors"],
        "em": metrics["em"],
        "f1": metrics["f1"],
        "bleu1": metrics["bleu1"],
        "per_question_type": metrics["per_question_type"],
    }

    # Write output
    os.makedirs(output_dir, exist_ok=True)

    details_path = os.path.join(output_dir, "details_judged.jsonl")
    with open(details_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = os.path.join(output_dir, "summary_judged.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Console summary
    bands = metrics["band_distribution"]
    print("\n" + "=" * 72)
    print("LLM JUDGE EVALUATION")
    print("=" * 72)
    print(f"Items judged:     {len(results)}")
    print(f"Judge model:      {judge_config.get('llm_model', '?')}")
    print(f"String accuracy:  {string_accuracy:.4f}")
    print(f"Judge score:      {metrics['judge_score']:.4f}  (mean across all items)")
    print(f"Errors:           {metrics['errors']}")
    print()
    print(f"Score distribution:")
    for band in ["1.0", "0.7", "0.5", "0.3", "0.0"]:
        n = bands.get(band, 0)
        bar = "█" * (n // max(1, len(results) // 50))
        print(f"  {band}: {n:>5d}  {bar}")
    print()
    print(f"EM:      {metrics['em']:.4f}")
    print(f"F1:      {metrics['f1']:.4f}")
    print(f"BLEU-1:  {metrics['bleu1']:.4f}")

    if metrics.get("per_question_type"):
        print(f"\n{'Question Type':<30s} {'Count':>5s}  {'Judge':>7s}  {'EM':>7s}")
        print("-" * 55)
        for qt, stats in metrics["per_question_type"].items():
            print(f"  {qt:<28s} {stats['count']:>5d}  {stats['judge_score']:>7.4f}  {stats['em']:>7.4f}")

    print("=" * 72)
    logger.success(f"Done in {elapsed:.1f}s")
    logger.info(f"Details  -> {details_path}")
    logger.info(f"Summary  -> {summary_path}")


if __name__ == "__main__":
    main()
