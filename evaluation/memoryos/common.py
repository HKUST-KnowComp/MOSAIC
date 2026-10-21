#!/usr/bin/env python3
"""Shared utilities for MemoryOS + MOSAIC evaluation.

Used by both build_memories.py and eval_questions.py.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any


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

    Build mode:   wrapper = MemoryOSEvalWrapper(uuid, api_key, config)
                  wrapper.add_conversation(...)  # sequential ingestion
                  wrapper.force_consolidation()  # build profile + knowledge
                  # state persists to disk automatically

    Query mode:   wrapper = MemoryOSEvalWrapper(uuid, api_key, config)
                  # state auto-loads from disk (if exists)
                  wrapper.query_eval(question, ...)  # read-only, concurrency-safe
    """

    def __init__(self, persona_uuid: str, api_key: str, config: dict):
        self.persona_uuid = persona_uuid
        mcfg = config["memoryos"]
        self._llm_model = mcfg["llm_model"]
        self._openai_base_url = mcfg.get("openai_base_url")
        self._extra_body = mcfg.get("extra_body")  # applied to ALL LLM calls
        self._data_storage_path = mcfg["data_storage_path"]

        self.memoryos = Memoryos(
            user_id=persona_uuid,
            openai_api_key=api_key,
            data_storage_path=self._data_storage_path,
            openai_base_url=self._openai_base_url,
            llm_model=self._llm_model,
            embedding_model_name=mcfg["embedding_model_name"],
            embedding_model_kwargs=mcfg.get("embedding_model_kwargs"),
            short_term_capacity=mcfg.get("short_term_capacity", 10),
            mid_term_capacity=mcfg.get("mid_term_capacity", 5000),
            mid_term_similarity_threshold=mcfg.get("mid_term_similarity_threshold", 0.6),
            mid_term_heat_threshold=mcfg.get("mid_term_heat_threshold", 5.0),
        )

        # Patch underlying OpenAI client so ALL LLM calls include extra_body
        if self._extra_body:
            _orig_create = self.memoryos.client.client.chat.completions.create
            _eb = self._extra_body

            def _patched_create(*args, **kwargs):
                kwargs.setdefault("extra_body", _eb)
                return _orig_create(*args, **kwargs)

            self.memoryos.client.client.chat.completions.create = _patched_create

        # Batch more pages per updater cycle — the multi-summary LLM call
        # (39% of ingestion cost) is shared per batch. Default drains 1 page;
        # with batch_size=10 it drains 10, cutting multi-summary calls 10x.
        _updater_batch = mcfg.get("updater_batch_size", 1)
        if _updater_batch > 1:
            _orig_process = self.memoryos.updater.process_short_term_to_mid_term
            _stm = self.memoryos.short_term_memory

            def _batched_process():
                original_cap = _stm.max_capacity
                target_cap = max(1, original_cap - _updater_batch)
                _stm.max_capacity = target_cap
                try:
                    _orig_process()
                finally:
                    _stm.max_capacity = original_cap

            self.memoryos.updater.process_short_term_to_mid_term = _batched_process

    # -- Build phase --

    def add_conversation(self, user_input: str, agent_response: str, timestamp: str = ""):
        """Feed a single QA pair into MemoryOS. Sequential per persona."""
        self.memoryos.add_memory(
            user_input=user_input,
            agent_response=agent_response,
            timestamp=timestamp if timestamp else None,
        )

    def force_consolidation(self):
        """Force mid-term analysis after ingestion."""
        self.memoryos.force_mid_term_analysis()

    # -- Query phase (read-only, safe for concurrent calls) --

    def query_eval(
        self,
        question: str,
        llm_cfg: dict,
        rcfg: dict,
        *,
        return_context: bool = False,
        question_type: str | None = None,
    ) -> str | tuple[str, dict]:
        """Query MemoryOS memory tiers. Read-only, concurrency-safe."""
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

        ku_rule = ""
        if str(question_type or "").strip().lower() == "knowledge_update":
            ku_rule = (
                "8. For knowledge_update questions (what changed / was updated), "
                'answer with the full transition as "from <old> to <new>". '
                "Do not answer with only the final/current state when both values "
                "appear in the evidence.\n"
            )
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
            f"{ku_rule}"
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
        predicted = answer.strip() if answer else ""
        if return_context:
            return predicted, {
                "retrieved_pages": retrieved_pages,
                "short_term_history": short_term_history,
                "retrieved_user_knowledge": retrieved_user_knowledge,
                "retrieved_assistant_knowledge": retrieved_assistant_knowledge,
                "user_profile": profile_text,
            }
        return predicted

    def storage_path(self) -> str:
        return os.path.join(self._data_storage_path, "users", self.persona_uuid)

    def cleanup(self):
        """Remove persisted memory files for this persona."""
        user_dir = self.storage_path()
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir, ignore_errors=True)
        asst_dir = os.path.join(self._data_storage_path, "assistants")
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


def count_all_pairs(filepath: str) -> int:
    """Count unique QA pairs across all sessions in a persona file."""
    data = load_json(filepath)
    seen_sessions: dict[str, dict] = {}
    for item in data.get("items", []):
        for session in item.get("haystack_sessions", []):
            sid = session.get("session_id", "")
            if sid and sid not in seen_sessions:
                seen_sessions[sid] = session

    total = 0
    for session in seen_sessions.values():
        total += len(convert_messages_to_qa_pairs(session.get("messages", [])))
    return total


def ingest_all_sessions(wrapper: MemoryOSEvalWrapper, persona_data: dict,
                        progress_callback=None) -> int:
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
            if progress_callback:
                progress_callback()

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
# Output helpers
# ---------------------------------------------------------------------------

def _collect_all_items(persona_results: list[dict]) -> list[dict]:
    all_items = []
    for pr in persona_results:
        if "items" in pr:
            all_items.extend(pr["items"])
    return all_items


def _print_summary(persona_results: list[dict], failed: list[dict], label: str = "EVALUATION"):
    all_items = _collect_all_items(persona_results)
    gm = compute_metrics(all_items)

    print("\n" + "=" * 60)
    print(f"{label} SUMMARY")
    print("=" * 60)
    print(f"Personas processed: {len(persona_results)}")
    print(f"Failed:            {len(failed)}")
    if gm.get("total_questions", 0) > 0:
        print(f"Total questions:   {gm.get('total_questions', 0)}")
        print(f"Overall accuracy:  {gm.get('overall_accuracy', 0):.3f}")
        print(f"  Answerable:      {gm.get('accuracy_answerable', 0):.3f} "
              f"({gm.get('correct_answerable', 0)} / {gm.get('answerable_questions', 0)})")
        print(f"  Unanswerable:    {gm.get('accuracy_unanswerable', 0):.3f} "
              f"({gm.get('correct_unanswerable', 0)} / {gm.get('unanswerable_questions', 0)})")
        print(f"\nPer question type:")
        for qtype, stats in gm.get("per_question_type", {}).items():
            print(f"  {qtype:25s}: {stats['accuracy']:.3f} ({stats['count']} items)")

    if failed:
        print(f"\nFAILED:")
        for f in failed:
            print(f"  {f.get('file', f.get('persona_uuid', 'unknown'))}: "
                  f"{f.get('error', 'unknown error')[:120]}")


def load_eval_config(config_path: str) -> dict:
    """Load and validate evaluation config."""
    _eval_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _eval_root not in sys.path:
        sys.path.insert(0, _eval_root)
    from env_bootstrap import (
        ensure_env_loaded,
        expand_mosaic_paths,
        resolve_api_key,
        resolve_base_url_for_api_key_env,
    )

    ensure_env_loaded()
    config = load_json(config_path)

    # Resolve API key
    api_key = config.get("api_key", "")
    api_key_env = config.get(
        "api_key_env",
        config.get("openai_api_key_env", "OPENAI_API_KEY"),
    )
    if not api_key:
        api_key = resolve_api_key(str(api_key_env))
    if not api_key:
        print("ERROR: No API key found. Set 'api_key' in config or set "
              f"{api_key_env} or LLM_API_KEY.")
        sys.exit(1)
    config["_resolved_api_key"] = api_key

    resolved_url = resolve_base_url_for_api_key_env(str(api_key_env))
    if resolved_url:
        for block_name in ("memoryos", "memory_config", "llm_config", "eval_config"):
            block = config.get(block_name)
            if isinstance(block, dict) and "openai_base_url" in block:
                block["openai_base_url"] = resolved_url

    return expand_mosaic_paths(config)
