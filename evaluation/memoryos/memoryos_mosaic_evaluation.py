#!/usr/bin/env python3
"""MOSAIC + MemoryOS evaluation script.

Two-phase architecture:
  Phase 1 (ingest): Build all persona memories in parallel (e.g. 5 workers).
                    Each persona ingests its sessions sequentially into MemoryOS.
  Phase 2 (eval):   Evaluate all 1000+ questions in parallel (e.g. 100 workers).
                    Each question queries its persona's pre-built memory (read-only, safe for concurrency).

Usage:
  cd /path/to/MOSAIC
  python evaluation/memoryos/memoryos_mosaic_evaluation.py --config evaluation/memoryos/memoryos_eval_config.json
  python evaluation/memoryos/memoryos_mosaic_evaluation.py --config evaluation/memoryos/memoryos_eval_config.json --persona-filter 0083bb89
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
import traceback
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import json


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


MEMORYOS_PATH = os.environ.get("MEMORYOS_PATH", "").strip()
if MEMORYOS_PATH and MEMORYOS_PATH not in sys.path:
    sys.path.insert(0, MEMORYOS_PATH)

try:
    from memoryos import Memoryos
except ImportError as exc:
    raise ImportError(
        "MemoryOS is not importable. Set MEMORYOS_PATH to the memoryos-pypi checkout."
    ) from exc

# ---------------------------------------------------------------------------
# MemoryOS evaluation wrapper
# ---------------------------------------------------------------------------

class MemoryOSEvalWrapper:
    """Wraps MemoryOS for clean evaluation without query contamination.

    MemoryOS.get_response() auto-adds the query and response back to memory.
    This wrapper uses MemoryOS for ingestion only, then queries via the
    internal retriever + LLM client directly (read-only, safe for concurrency).
    """

    def __init__(self, persona_uuid: str, api_key: str, config: dict):
        self.persona_uuid = persona_uuid
        mcfg = config["memoryos"]
        self._llm_model = mcfg["llm_model"]
        self._openai_base_url = mcfg.get("openai_base_url")
        self._extra_body = mcfg.get("extra_body")  # applied to ALL LLM calls

        self.memoryos = Memoryos(
            user_id=persona_uuid,
            openai_api_key=api_key,
            data_storage_path=mcfg["data_storage_path"],
            openai_base_url=self._openai_base_url,
            llm_model=self._llm_model,
            embedding_model_name=mcfg["embedding_model_name"],
            short_term_capacity=mcfg["short_term_capacity"],
            mid_term_capacity=mcfg["mid_term_capacity"],
            mid_term_similarity_threshold=mcfg["mid_term_similarity_threshold"],
            mid_term_heat_threshold=mcfg["mid_term_heat_threshold"],
        )

        # Patch the underlying OpenAI client so ALL LLM calls (internal
        # MemoryOS operations AND eval queries) include extra_body.  This is
        # needed for reasoning models where thinking must be disabled.
        if self._extra_body:
            _orig_create = self.memoryos.client.client.chat.completions.create
            _extra_body = self._extra_body

            def _patched_create(*args, **kwargs):
                kwargs.setdefault("extra_body", _extra_body)
                return _orig_create(*args, **kwargs)

            self.memoryos.client.client.chat.completions.create = _patched_create

    def add_conversation(self, user_input: str, agent_response: str, timestamp: str = ""):
        """Feed a single QA pair into MemoryOS memory (sequential per persona)."""
        self.memoryos.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp if timestamp else None,
        )

    def query_eval(self, question: str, llm_cfg: dict, rcfg: dict) -> str:
        """Query MemoryOS memory tiers. Read-only, safe for concurrent calls."""
        retrieval = self.memoryos.retriever.retrieve_context(
            user_query=question,
            user_id=self.persona_uuid,
            segment_similarity_threshold=rcfg["segment_similarity_threshold"],
            page_similarity_threshold=rcfg["page_similarity_threshold"],
            knowledge_threshold=rcfg["knowledge_threshold"],
            top_k_sessions=rcfg["top_k_sessions"],
            top_k_knowledge=rcfg["top_k_knowledge"],
        )

        retrieved_pages = retrieval["retrieved_pages"]
        retrieved_user_knowledge = retrieval["retrieved_user_knowledge"]
        retrieved_assistant_knowledge = retrieval["retrieved_assistant_knowledge"]

        # Short-term history
        short_term_history = self.memoryos.short_term_memory.get_all()
        history_lines = []
        for qa in short_term_history:
            history_lines.append(
                f"User: {qa.get('user_input', '')}\n"
                f"Assistant: {qa.get('agent_response', '')}\n"
                f"Time: {qa.get('timestamp', '')}"
            )
        history_text = "\n".join(history_lines) if history_lines else "(no recent history)"

        # Retrieved mid-term pages
        retrieval_lines = []
        for page in retrieved_pages:
            retrieval_lines.append(
                f"User: {page.get('user_input', '')}\n"
                f"Assistant: {page.get('agent_response', '')}\n"
                f"Time: {page.get('timestamp', '')}\n"
                f"Summary: {page.get('meta_info', 'N/A')}"
            )
        retrieval_text = "\n\n".join(retrieval_lines) if retrieval_lines else "(no relevant past conversations)"

        # User profile
        profile_text = self.memoryos.user_long_term_memory.get_raw_user_profile(self.persona_uuid)
        if not profile_text or profile_text.lower() in ("none", ""):
            profile_text = "No detailed profile available."

        # User knowledge
        uk_lines = [f"- {kn.get('knowledge', str(kn))}" for kn in retrieved_user_knowledge]
        user_knowledge_text = "\n".join(uk_lines) if uk_lines else "(none)"

        # Assistant knowledge
        ak_lines = [f"- {ak.get('knowledge', str(ak))}" for ak in retrieved_assistant_knowledge]
        assistant_knowledge_text = "\n".join(ak_lines) if ak_lines else "(none)"

        system_prompt = (
            "You are an evaluation assistant. Your task is to answer questions about a user "
            "based on conversation history and retrieved memories.\n\n"
            f"=== User Profile ===\n{profile_text}\n\n"
            f"=== User Knowledge ===\n{user_knowledge_text}\n\n"
            f"=== Assistant Knowledge ===\n{assistant_knowledge_text}\n\n"
            "Answer rules:\n"
            "1. Answer with ONLY the requested fact or entity. Be extremely concise.\n"
            '2. Example: "What did the charity race raise awareness for?" -> "mental health"\n'
            '3. Example: "What time did work start?" -> "8:15 AM"\n'
            "4. For dates, use format like '15 July 2023' or 'April 2026'.\n"
            "5. For durations, use '3 months' or '2 years'.\n"
            "6. If the question cannot be answered from the provided information, respond with exactly: UNANSWERABLE\n"
            "7. Do NOT include explanations, just the answer.\n"
        )

        user_prompt = (
            "<RECENT CONVERSATION>\n"
            f"{history_text}\n\n"
            "<RETRIEVED MEMORIES>\n"
            f"{retrieval_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.memoryos.client.chat_completion(
            model=llm_cfg.get("eval_model", self._llm_model),
            messages=messages,
            temperature=llm_cfg.get("temperature", 0.0),
            max_tokens=llm_cfg.get("max_tokens", 200),
        )
        return answer.strip() if answer else ""

    def force_consolidation(self):
        """Force mid-term analysis to build profiles and knowledge base after ingestion."""
        self.memoryos.force_mid_term_analysis()

    def cleanup(self):
        """Remove persisted memory files for this persona."""
        user_dir = os.path.join(self.memoryos.data_storage_path, "users", self.persona_uuid)
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir, ignore_errors=True)
        asst_dir = os.path.join(self.memoryos.data_storage_path, "assistants")
        if os.path.exists(asst_dir):
            shutil.rmtree(asst_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------

def convert_messages_to_qa_pairs(messages: list[dict]) -> list[dict]:
    """Convert alternating user/assistant messages to add_memory format.

    Multimodal content is textually described via the 'textualized' field.
    """
    pairs = []
    current_user_msg = None

    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")

        mm = msg.get("multimodal_content")
        if mm:
            textualized = mm.get("textualized", "")
            if textualized:
                modality = mm.get("modality", "media")
                content = f"{content}\n[{modality}: {textualized}]"
            elif mm.get("natural_utterance"):
                modality = mm.get("modality", "media")
                content = f"{content}\n[{modality}: {mm['natural_utterance']}]"

        if role == "user":
            current_user_msg = content
        elif role == "assistant":
            pairs.append({
                "user_input": current_user_msg or "",
                "agent_response": content,
            })
            current_user_msg = None

    return pairs


def ingest_all_sessions(wrapper: MemoryOSEvalWrapper, persona_data: dict) -> int:
    """Deduplicate and ingest all conversation sessions for a persona.

    Returns count of QA pairs ingested.
    """
    seen_sessions: dict[str, dict] = OrderedDict()

    for item in persona_data.get("items", []):
        for session in item.get("haystack_sessions", []):
            sid = session.get("session_id", "")
            if sid and sid not in seen_sessions:
                seen_sessions[sid] = session

    def _sort_key(s: dict) -> str:
        return s.get("timestamp", s.get("session_date", ""))

    sorted_sessions = sorted(seen_sessions.values(), key=_sort_key)

    total_pairs = 0
    for session in sorted_sessions:
        timestamp = session.get("timestamp", session.get("session_date", ""))
        pairs = convert_messages_to_qa_pairs(session.get("messages", []))
        for pair in pairs:
            wrapper.add_conversation(
                user_input=pair["user_input"],
                agent_response=pair["agent_response"],
                timestamp=timestamp,
            )
            total_pairs += 1

    wrapper.force_consolidation()
    return total_pairs


# ---------------------------------------------------------------------------
# Answer evaluation
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    text = text.strip().lower()
    for prefix in ["answer:", "the answer is:", "response:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return " ".join(text.split())


def evaluate_answer(predicted: str, ground_truth: str,
                    answer_aliases: list[str]) -> dict:
    norm_pred = _normalize_answer(predicted)
    norm_truth = _normalize_answer(ground_truth)
    norm_aliases = [_normalize_answer(a) for a in (answer_aliases or [])]

    exact_match = norm_pred == norm_truth
    alias_match = exact_match or any(norm_pred == a for a in norm_aliases)

    return {
        "exact_match": exact_match,
        "alias_match": alias_match,
        "normalized_predicted": norm_pred,
        "normalized_ground_truth": norm_truth,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(items: list[dict]) -> dict:
    total = len(items)
    if total == 0:
        return {"total_questions": 0}

    answerable = [r for r in items if r.get("answerability") == "answerable"]
    unanswerable = [r for r in items if r.get("answerability") == "unanswerable"]

    correct_answerable = sum(1 for r in answerable if r["evaluation"]["alias_match"])
    correct_unanswerable = sum(
        1 for r in unanswerable if r["evaluation"].get("unanswerable_correct", False)
    )

    by_type: dict[str, list] = defaultdict(list)
    for r in items:
        by_type[r.get("question_type", "unknown")].append(r)

    per_type = {}
    for qtype, type_items in by_type.items():
        ta = [r for r in type_items if r.get("answerability") == "answerable"]
        tu = [r for r in type_items if r.get("answerability") == "unanswerable"]
        tc = sum(1 for r in ta if r["evaluation"]["alias_match"]) \
           + sum(1 for r in tu if r["evaluation"].get("unanswerable_correct", False))
        per_type[qtype] = {
            "count": len(type_items),
            "answerable": len(ta),
            "unanswerable": len(tu),
            "accuracy": tc / len(type_items) if type_items else 0,
        }

    return {
        "total_questions": total,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(unanswerable),
        "correct_answerable": correct_answerable,
        "correct_unanswerable": correct_unanswerable,
        "accuracy_answerable": correct_answerable / len(answerable) if answerable else 0,
        "accuracy_unanswerable": correct_unanswerable / len(unanswerable) if unanswerable else 0,
        "overall_accuracy": (correct_answerable + correct_unanswerable) / total if total else 0,
        "per_question_type": per_type,
    }


# ---------------------------------------------------------------------------
# Phase 1: Build memories
# ---------------------------------------------------------------------------

def _build_memory(filepath: str, api_key: str, config: dict) -> dict:
    """Build MemoryOS instance for one persona. Returns ingest result dict."""
    persona_data = load_json(filepath)
    persona = persona_data.get("persona", {})
    persona_uuid = persona.get("uuid", os.path.basename(filepath))
    persona_name = persona.get("name", "unknown")
    items = persona_data.get("items", [])

    wrapper = MemoryOSEvalWrapper(persona_uuid, api_key, config)

    start = time.time()
    num_pairs = ingest_all_sessions(wrapper, persona_data)
    elapsed = time.time() - start

    return {
        "persona_uuid": persona_uuid,
        "persona_name": persona_name,
        "file": os.path.basename(filepath),
        "num_pairs_ingested": num_pairs,
        "ingest_time_seconds": round(elapsed, 1),
        "num_items": len(items),
        "wrapper": wrapper,
        "items": items,
    }


# ---------------------------------------------------------------------------
# Phase 2: Evaluate questions
# ---------------------------------------------------------------------------

def _eval_one_question(item: dict, wrapper: MemoryOSEvalWrapper,
                       llm_cfg: dict, rcfg: dict) -> dict:
    """Evaluate a single question. Returns item result dict (no wrapper reference)."""
    question = item.get("question", "")
    answerability = item.get("answerability", "answerable")
    ground_truth = item.get("answer", "")
    aliases = item.get("answer_aliases", [])

    start = time.time()
    try:
        predicted = wrapper.query_eval(question, llm_cfg, rcfg)
    except Exception as e:
        predicted = f"ERROR: {e}"
    latency = time.time() - start

    evaluation = evaluate_answer(predicted, ground_truth, aliases)
    if answerability == "unanswerable":
        evaluation["unanswerable_correct"] = "unanswerable" in predicted.lower()

    return {
        "question_id": item.get("question_id", ""),
        "question_type": item.get("question_type", "unknown"),
        "answerability": answerability,
        "question": question,
        "ground_truth_answer": ground_truth,
        "answer_aliases": aliases,
        "predicted_answer": predicted,
        "evaluation": evaluation,
        "latency_seconds": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _collect_all_items(persona_results: list[dict]) -> list[dict]:
    all_items = []
    for pr in persona_results:
        if "items" in pr:
            all_items.extend(pr["items"])
    return all_items


def _write_results(output_dir: str, persona_results: list[dict],
                   failed_builds: list[dict], config: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "memoryos_mosaic_results.json")

    all_items = _collect_all_items(persona_results)
    global_metrics = compute_metrics(all_items)

    # Strip wrapper refs from output (not serializable)
    clean_personas = []
    for pr in persona_results:
        cp = {k: v for k, v in pr.items() if k not in ("wrapper",)}
        clean_personas.append(cp)

    report = {
        "config_summary": {
            "data_dir": config.get("data_dir", ""),
            "memoryos_model": config.get("memoryos", {}).get("llm_model", ""),
            "eval_model": config.get("llm", {}).get("eval_model", ""),
            "ingest_workers": config.get("ingest_workers", 5),
            "eval_workers": config.get("eval_workers", 100),
        },
        "total_personas_processed": len(persona_results),
        "total_personas_failed": len(failed_builds),
        "total_questions": len(all_items),
        "global_metrics": global_metrics,
        "personas": clean_personas,
        "failed_builds": failed_builds,
        "generated_at": datetime.now().isoformat(),
    }
    save_json(output_path, report)
    return output_path


def _print_summary(persona_results: list[dict], failed_builds: list[dict]):
    all_items = _collect_all_items(persona_results)
    gm = compute_metrics(all_items)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Personas processed: {len(persona_results)}")
    print(f"Personas failed:   {len(failed_builds)}")
    print(f"Total questions:   {gm.get('total_questions', 0)}")
    print(f"Overall accuracy:  {gm.get('overall_accuracy', 0):.3f}")
    print(f"  Answerable:      {gm.get('accuracy_answerable', 0):.3f} "
          f"({gm.get('correct_answerable', 0)} / {gm.get('answerable_questions', 0)})")
    print(f"  Unanswerable:    {gm.get('accuracy_unanswerable', 0):.3f} "
          f"({gm.get('correct_unanswerable', 0)} / {gm.get('unanswerable_questions', 0)})")
    print(f"\nPer question type:")
    for qtype, stats in gm.get("per_question_type", {}).items():
        print(f"  {qtype:25s}: {stats['accuracy']:.3f} ({stats['count']} items)")

    if failed_builds:
        print(f"\nFAILED BUILDS:")
        for f in failed_builds:
            print(f"  {f.get('file', 'unknown')}: {f.get('error', 'unknown error')[:120]}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate MemoryOS on the MOSAIC filtered benchmark dataset"
    )
    p.add_argument("--config", required=True,
                   help="Path to evaluation config JSON")
    p.add_argument("--ingest-workers", type=int, default=None,
                   help="Parallel workers for Phase 1 memory construction (default from config)")
    p.add_argument("--eval-workers", type=int, default=None,
                   help="Parallel workers for Phase 2 QA evaluation (default from config)")
    p.add_argument("--resume", default=None,
                   help="Path to previous results JSON to skip already-processed personas")
    p.add_argument("--persona-filter", default=None,
                   help="Comma-separated persona UUID substrings (for testing subsets)")
    return p.parse_args()


def main():
    args = parse_args()
    _eval_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _eval_root not in sys.path:
        sys.path.insert(0, _eval_root)
    from env_bootstrap import expand_mosaic_paths

    config = expand_mosaic_paths(load_json(args.config))

    # Resolve API key
    api_key = config.get("api_key", "")
    if not api_key:
        api_key_env = config.get("openai_api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env, "")
    if not api_key:
        print("ERROR: No API key found. Set 'api_key' in config or set env var "
              f"{config.get('openai_api_key_env', 'OPENAI_API_KEY')}.")
        sys.exit(1)

    data_dir = config["data_dir"]
    file_glob = config.get("persona_file_glob", "stage_2_text_based_*.json")
    persona_files = sorted(glob.glob(os.path.join(data_dir, file_glob)))

    if not persona_files:
        print(f"ERROR: No files matching '{file_glob}' found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(persona_files)} persona files")

    # Resume
    resume_uuids: set[str] = set()
    existing_results: list[dict] = []
    if args.resume and os.path.exists(args.resume):
        prev = load_json(args.resume)
        for pr in prev.get("personas", []):
            resume_uuids.add(pr.get("persona_uuid", ""))
        existing_results = prev.get("personas", [])
        print(f"Resuming: {len(resume_uuids)} personas already processed")

    # Persona filter
    if args.persona_filter:
        filters = [f.strip() for f in args.persona_filter.split(",")]
        persona_files = [pf for pf in persona_files
                         if any(filt in os.path.basename(pf) for filt in filters)]
        print(f"Filtered to {len(persona_files)} persona files")

    # Worker counts
    ingest_workers = args.ingest_workers or config.get("ingest_workers", 5)
    eval_workers = args.eval_workers or config.get("eval_workers", 100)
    output_dir = config.get("output_dir", "evaluation/memoryos/results/")

    # Ensure MemoryOS temp dir exists
    os.makedirs(config["memoryos"]["data_storage_path"], exist_ok=True)

    # Filter out already-processed personas
    pending_files = []
    for pf in persona_files:
        try:
            data = load_json(pf)
            puid = data.get("persona", {}).get("uuid", "")
        except Exception:
            puid = ""
        if puid in resume_uuids:
            print(f"Skipping {puid[:12]}... (already in resume output)")
            continue
        pending_files.append(pf)

    total_personas = len(persona_files)
    pending_count = len(pending_files)
    print(f"\n{'=' * 60}")
    print(f"PHASE 1: Building memories for {pending_count} persona(s)")
    print(f"  Workers: {ingest_workers}")
    print(f"{'=' * 60}")

    if not pending_files:
        _write_results(output_dir, list(existing_results), [], config)
        _print_summary(list(existing_results), [])
        return

    # ---- Phase 1: Build all memories in parallel ----
    memory_pool: dict[str, MemoryOSEvalWrapper] = {}  # persona_uuid -> wrapper
    persona_meta: dict[str, dict] = {}  # persona_uuid -> {name, file, num_pairs, items: [...]}
    failed_builds: list[dict] = []

    build_start = time.time()
    with ThreadPoolExecutor(max_workers=ingest_workers) as executor:
        future_map = {}
        for pf in pending_files:
            future = executor.submit(_build_memory, pf, api_key, config)
            future_map[future] = pf

        completed = 0
        for future in as_completed(future_map):
            pf = future_map[future]
            completed += 1
            try:
                result = future.result()
                puid = result["persona_uuid"]
                memory_pool[puid] = result.pop("wrapper")
                persona_meta[puid] = result
                print(f"[build {completed}/{pending_count}] DONE: {result['persona_name'][:30]:30s} "
                      f"{result['num_pairs_ingested']} QA pairs in {result['ingest_time_seconds']}s")
            except Exception as e:
                failed_builds.append({"file": pf, "error": traceback.format_exc()})
                print(f"[build {completed}/{pending_count}] FAILED: {os.path.basename(pf)} -- {e}")

    build_elapsed = time.time() - build_start
    print(f"\nPhase 1 complete: {len(memory_pool)} memories built, "
          f"{len(failed_builds)} failed ({build_elapsed:.0f}s)")

    if not memory_pool:
        print("ERROR: No memories built successfully. Exiting.")
        sys.exit(1)

    # ---- Phase 2: Evaluate all questions in parallel ----
    # Flatten: collect all (item, persona_uuid) tuples
    all_work: list[tuple[dict, str]] = []
    for puid, meta in persona_meta.items():
        for item in meta["items"]:
            all_work.append((item, puid))

    print(f"\n{'=' * 60}")
    print(f"PHASE 2: Evaluating {len(all_work)} questions")
    print(f"  Workers: {eval_workers}")
    print(f"{'=' * 60}")

    rcfg = config["retrieval"]
    llm_cfg = config.get("llm", {})

    # Per-persona result accumulator (thread-safe via GIL for list append)
    persona_item_results: dict[str, list[dict]] = defaultdict(list)

    eval_start = time.time()
    eval_completed = 0
    eval_correct = 0

    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        future_map = {}
        for item, puid in all_work:
            wrapper = memory_pool[puid]
            future = executor.submit(_eval_one_question, item, wrapper, llm_cfg, rcfg)
            future_map[future] = (item, puid)

        for future in as_completed(future_map):
            item, puid = future_map[future]
            eval_completed += 1
            try:
                item_result = future.result()
                persona_item_results[puid].append(item_result)
                if item_result["evaluation"]["alias_match"] or \
                   item_result["evaluation"].get("unanswerable_correct", False):
                    eval_correct += 1

                if eval_completed % 100 == 0 or eval_completed == len(all_work):
                    elapsed = time.time() - eval_start
                    rate = eval_completed / elapsed if elapsed > 0 else 0
                    print(f"[eval {eval_completed}/{len(all_work)}] "
                          f"acc={eval_correct / eval_completed:.3f} "
                          f"({rate:.0f} q/s, {elapsed:.0f}s)")

            except Exception as e:
                print(f"[eval {eval_completed}/{len(all_work)}] ERROR: {item.get('question_id', '?')} -- {e}")

    eval_elapsed = time.time() - eval_start
    print(f"\nPhase 2 complete: {eval_completed} questions in {eval_elapsed:.0f}s "
          f"({eval_completed / eval_elapsed:.1f} q/s)")

    # ---- Assemble results per persona ----
    all_persona_results = list(existing_results)  # from resume
    for puid, meta in persona_meta.items():
        items = persona_item_results.get(puid, [])
        all_persona_results.append({
            "persona_uuid": puid,
            "persona_name": meta["persona_name"],
            "file": meta["file"],
            "ingest_time_seconds": meta["ingest_time_seconds"],
            "num_qa_pairs_ingested": meta["num_pairs_ingested"],
            "metrics": compute_metrics(items),
            "items": items,
        })

    # ---- Phase 3: Cleanup ----
    print(f"\nCleaning up {len(memory_pool)} memory instances...")
    for wrapper in memory_pool.values():
        wrapper.cleanup()

    # ---- Final output ----
    output_path = _write_results(output_dir, all_persona_results, failed_builds, config)
    _print_summary(all_persona_results, failed_builds)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
