#!/usr/bin/env python3
"""Shared utilities for A-MEM + Mosaic evaluation scripts.

Used by both ``build_index.py`` and ``query_index.py``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
from collections import defaultdict
from datetime import datetime

import numpy as np
from loguru import logger

# Project root (evaluation/amem/ -> 3 levels up)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TIMESTAMP_FMT = "%Y/%m/%d (%a) %H:%M"
AMEM_TIMESTAMP_FMT  = "%Y%m%d%H%M"
FALLBACK_DATE_FMT   = "%Y/%m/%d"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load JSON config, resolving API keys.

    Supports both ``"api_key": "sk-..."`` (direct value) and
    ``"api_key_env": "OPENAI_API_KEY"`` (env-var lookup).  Direct value wins.
    Loads the repo ``.env`` on first use.
    """
    import sys

    _eval_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _eval_root not in sys.path:
        sys.path.insert(0, _eval_root)
    from env_bootstrap import (
        ensure_env_loaded,
        expand_mosaic_paths,
        resolve_base_url_for_api_key_env,
    )

    ensure_env_loaded()

    with open(config_path) as f:
        config = json.load(f)

    def _resolve_key(cfg: dict):
        env_var = cfg.get("api_key_env")
        if not cfg.get("api_key") and env_var:
            cfg["api_key"] = os.getenv(env_var, "")
        if env_var:
            resolved = resolve_base_url_for_api_key_env(str(env_var))
            if resolved and "base_url" in cfg:
                cfg["base_url"] = resolved

    _resolve_key(config.get("amem_config", {}))

    qa = config.get("qa_evaluation", {})
    if qa.get("enabled") and "llm_config" in qa:
        _resolve_key(qa["llm_config"])

    return expand_mosaic_paths(config)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_persona_file(path: str) -> tuple[dict, list[dict]]:
    """Load a single persona JSON file.  Returns *(persona, items)*."""
    with open(path) as f:
        data = json.load(f)
    return data["persona"], data["items"]


# ---------------------------------------------------------------------------
# Shared embedding model  (singleton — one copy regardless of worker count)
# ---------------------------------------------------------------------------
#
# Strategy: monkey-patch ``sentence_transformers.SentenceTransformer`` at the
# class level so that EVERY ``SentenceTransformerEmbeddingFunction`` created
# anywhere (A-MEM init, PersistentChromaRetriever, …) receives the **same**
# underlying model object.  A mutex serialises ``encode()`` calls so
# concurrent worker threads never touch the model simultaneously.

_shared_st_model: object | None = None
_st_lock = threading.Lock()
_encode_lock = threading.Lock()


def _install_st_singleton(model_name: str, device: str):
    """Ensure the SentenceTransformer singleton is loaded.

    Idempotent & thread-safe — only the first caller loads weights.
    """
    global _shared_st_model
    if _shared_st_model is not None:
        return

    with _st_lock:
        if _shared_st_model is not None:
            return

        from sentence_transformers import SentenceTransformer as _OrigST

        _shared_st_model = _OrigST(model_name, device=device)
        logger.info(
            f"Loaded shared SentenceTransformer '{model_name}' on '{device}'"
        )

        # Replace the class so every future instantiation returns the cached
        # model.  __new__ returns the shared instance; __init__ is a no-op
        # so tqdm "Loading weights" doesn't fire on every instantiation.
        class _CachedST:
            def __new__(cls, *args, **kwargs):
                return _shared_st_model
            def __init__(self, *args, **kwargs):
                pass

        import sentence_transformers as _st_mod
        _st_mod.SentenceTransformer = _CachedST


def _repair_json(text: str) -> str:
    """Attempt to fix common LLM JSON issues (unescaped control chars)."""
    import re
    # Replace literal newlines, tabs, carriage-returns inside JSON strings
    # with their escaped equivalents.  We look inside double-quoted strings.
    def _escape_controls(match):
        s = match.group(1)
        s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return '"' + s + '"'

    text = re.sub(r'"([^"]*)"', _escape_controls, text)
    return text


def _wrap_retriever_ef(retriever):
    """Wrap a retriever's embedding-function in a thread-safe mutex proxy.

    Patches both ``retriever.embedding_function`` and the internal ChromaDB
    collection reference so every ``add`` / ``query`` goes through the lock.
    """
    delegate = retriever.embedding_function

    class _LockedEF:
        def __call__(self_, input):
            with _encode_lock:
                return delegate(input)

        def __getattr__(self_, name):
            return getattr(delegate, name)

    locked = _LockedEF()
    retriever.embedding_function = locked
    coll = retriever.collection
    for attr in ("_embedding_function", "embedding_function"):
        try:
            if hasattr(coll, attr):
                setattr(coll, attr, locked)
        except Exception:
            pass


# ---- work around ChromaDB 1.5.x tenant error (module level, once) ----
# A-MEM's __init__ creates an ephemeral ChromaRetriever which calls
# ``chromadb.Client()``.  In ChromaDB >= 1.x this raises "Could not
# connect to tenant default_tenant".  We patch ChromaRetriever once at
# import time to fall back to a temp PersistentClient.
import agentic_memory.retrievers as _amem_ret_mod
import chromadb as _chromadb
import tempfile as _tempfile

_orig_cr_init = _amem_ret_mod.ChromaRetriever.__init__

def _safe_cr_init(self_, *args, **kw):
    self_._tmp_dir = None
    try:
        _orig_cr_init(self_, *args, **kw)
    except Exception:
        import shutil as _shutil
        _td = _tempfile.mkdtemp(prefix="amem_tmp_cr_")
        self_._tmp_dir = _td
        self_.client = _chromadb.PersistentClient(path=_td)
        model_name = kw.get("model_name", "all-MiniLM-L6-v2")
        collection_name = kw.get("collection_name", "memories")
        self_.embedding_function = (
            _amem_ret_mod.SentenceTransformerEmbeddingFunction(
                model_name=model_name,
            )
        )
        self_.collection = self_.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self_.embedding_function,
        )
    self_.client.reset = lambda: None  # PersistentClient raises on reset

_amem_ret_mod.ChromaRetriever.__init__ = _safe_cr_init


# ---------------------------------------------------------------------------
# A-MEM instance
# ---------------------------------------------------------------------------


def build_amem_instance(amem_config: dict, persist_dir: str | None = None, *, extend: bool = False):
    """Create an AgenticMemorySystem, optionally with a persistent backend.

    The embedding model is a process-wide singleton — even with
    ``build_workers > 1``, only **one** copy of the weights is loaded.

    Parameters
    ----------
    amem_config:
        Must contain *model_name*, *llm_backend*, *llm_model*.
        Optional: *evolution_enabled*, *evo_threshold*, *api_key*,
        *embedding_device* (e.g. ``"cuda:0"``, ``"cpu"``).
    persist_dir:
        If given, the ChromaDB retriever is swapped for a
        ``PersistentChromaRetriever`` stored at this path.  ``extend=True``
        is used so an existing collection can be re-opened.
    """
    from agentic_memory.memory_system import AgenticMemorySystem

    evolution_enabled = amem_config.get("evolution_enabled", True)
    api_key = amem_config.get("api_key")
    device = amem_config.get("embedding_device", "cpu")
    model_name = amem_config["model_name"]
    base_url = amem_config.get("base_url")

    if not evolution_enabled and not api_key:
        api_key = "sk-placeholder"

    # Install the SentenceTransformer singleton BEFORE A-MEM creates its
    # first ChromaRetriever (which would otherwise load its own copy).
    _install_st_singleton(model_name, device)

    system = AgenticMemorySystem(
        model_name=model_name,
        llm_backend=amem_config["llm_backend"],
        llm_model=amem_config["llm_model"],
        evo_threshold=amem_config.get("evo_threshold", 100),
        api_key=api_key,
    )

    # Wrap the ephemeral retriever so encode calls go through the shared lock.
    _wrap_retriever_ef(system.retriever)

    # ---- swap in persistent retriever if requested ----
    if persist_dir is not None:
        # Clean up temp dir from the ephemeral retriever (created by _safe_cr_init)
        import shutil as _shutil
        if getattr(system.retriever, "_tmp_dir", None):
            _shutil.rmtree(system.retriever._tmp_dir, ignore_errors=True)

        from agentic_memory.retrievers import PersistentChromaRetriever
        os.makedirs(persist_dir, exist_ok=True)
        system.retriever = PersistentChromaRetriever(
            directory=persist_dir,
            collection_name="memories",
            model_name=model_name,
            extend=extend,   # False=build (fail if exists), True=query (reconnect)
        )
        _wrap_retriever_ef(system.retriever)
        logger.debug(f"Persistent ChromaDB at {persist_dir}")

    # ---- patch LLM client for custom base_url (DeepSeek, etc.) ----
    if evolution_enabled and base_url:
        from openai import OpenAI
        system.llm_controller.llm.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.debug(f"LLM client base_url -> {base_url}")

    # ---- patch get_completion for DeepSeek compat + thinking ----
    if evolution_enabled:
        _thinking_disabled = amem_config.get("thinking_disabled", False)

        # A-MEM passes json_schema but DeepSeek only accepts json_object.
        def _fix_rf(rf: dict) -> dict:
            if rf and rf.get("type") == "json_schema":
                return {"type": "json_object"}
            return rf

        import types

        def _patched_get_completion(
            self_, prompt, response_format=None, temperature=0.7,
        ):
            rf = _fix_rf(response_format)
            kwargs = dict(
                model=self_.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You must respond with a single valid JSON object. "
                            "Escape all double-quotes, backslashes, and control "
                            "characters inside string values. Do not truncate."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=rf,
                temperature=temperature,
                max_tokens=8192,
            )
            if _thinking_disabled:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = self_.client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content

            # DeepSeek's json_object mode doesn't enforce valid escaping;
            # fix common issues so json.loads() in process_memory succeeds.
            try:
                json.loads(text)
            except json.JSONDecodeError:
                text = _repair_json(text)
            return text

        system.llm_controller.llm.get_completion = types.MethodType(
            _patched_get_completion, system.llm_controller.llm,
        )
        logger.debug(
            f"LLM get_completion patched "
            f"(json_schema->json_object, thinking_disabled={_thinking_disabled})"
        )

    if not evolution_enabled:
        system.process_memory = lambda note: (False, note)
        logger.debug("Memory evolution DISABLED")

    return system


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def format_message(msg: dict, session_timestamp_str: str) -> tuple[str, str]:
    """Convert a Mosaic message into *(formatted_content, amem_timestamp)*.

    For multimodal messages (image/video/audio), only the textualized
    description is embedded — not the conversational wrapper text.  For
    text-only messages, the message content is embedded as-is.
    """
    role = msg.get("role", "user")
    content = msg.get("content", "")
    multimodal = msg.get("multimodal_content")

    if multimodal and multimodal.get("type") is not None:
        # image / video / audio: embed only the textualized description
        textualized = multimodal.get("textualized", "")
        full_content = textualized if textualized else content
    else:
        # text-only message: embed content as-is
        full_content = content

    ts_str = (session_timestamp_str or "").strip()
    try:
        dt = datetime.strptime(ts_str, SESSION_TIMESTAMP_FMT)
    except (ValueError, TypeError):
        try:
            date_only = ts_str.split(" (")[0]
            dt = datetime.strptime(date_only, FALLBACK_DATE_FMT)
        except (ValueError, TypeError):
            dt = datetime(2020, 1, 1, 12, 0)

    amem_ts = dt.strftime(AMEM_TIMESTAMP_FMT)
    formatted = f"[{role}] {full_content}"
    return formatted, amem_ts


# ---------------------------------------------------------------------------
# Memory population
# ---------------------------------------------------------------------------


def populate_all_messages(items: list[dict], system, pbar=None) -> dict:
    """Add every message from every item's *haystack_sessions* into A-MEM.

    Sessions are deduplicated by *session_id* (filler sessions can be shared
    across items) and sorted by timestamp so memories are ingested in
    chronological order.

    Returns *note_registry* --- ``note_id -> {fact_key, has_answer, item_id, ...}``.
    """
    # ---- deduplicate sessions & map to first item ----
    seen: dict[str, dict] = {}
    session_item_map: dict[str, str] = {}
    for item in items:
        item_id = item.get("question_id", "unknown")
        for session in item.get("haystack_sessions", []):
            sid = session.get("session_id", "")
            if sid and sid not in seen:
                seen[sid] = session
                session_item_map[sid] = item_id

    # ---- sort by timestamp (chronological ingestion) ----
    def _sort_key(s: dict) -> str:
        return s.get("timestamp", s.get("session_date", ""))

    sorted_sessions = sorted(seen.values(), key=_sort_key)

    total_msgs = sum(len(s.get("messages", [])) for s in sorted_sessions)
    logger.info(
        f"Ingesting {total_msgs} messages across {len(sorted_sessions)} sessions ..."
    )

    # ---- ingest messages once per unique session ----
    note_registry: dict[str, dict] = {}
    ingested = 0
    log_every = max(1, total_msgs // 20)  # log at ~5 % intervals

    for session in sorted_sessions:
        session_id = session.get("session_id", "unknown")
        session_ts = session.get("timestamp") or session.get("session_date", "")
        item_id = session_item_map.get(session_id, "unknown")

        for msg in session.get("messages", []):
            formatted_text, amem_ts = format_message(msg, session_ts)

            note_id = system.add_note(
                content=formatted_text,
                time=amem_ts,
                tags=[session_id, item_id],
                category=session_id,
            )

            note_registry[note_id] = {
                "fact_key":   msg.get("fact_key"),
                "has_answer": msg.get("has_answer"),
                "item_id":    item_id,
                "session_id": session_id,
                "role":       msg.get("role"),
                "timestamp":  amem_ts,
            }

            ingested += 1
            if pbar is not None:
                pbar.update(1)
            if ingested % log_every == 0 or ingested == total_msgs:
                logger.debug(f"Ingested {ingested}/{total_msgs} messages")

    return note_registry


# ---------------------------------------------------------------------------
# Serialization helpers (build -> disk, query <- disk)
# ---------------------------------------------------------------------------


def memories_to_dicts(system) -> list[dict]:
    """Serialize ``system.memories`` (MemoryNote objects) to plain dicts."""
    return [
        {
            "id":                m.id,
            "content":           m.content,
            "keywords":          m.keywords,
            "links":             m.links,
            "retrieval_count":   m.retrieval_count,
            "timestamp":         m.timestamp,
            "last_accessed":     m.last_accessed,
            "context":           m.context,
            "evolution_history": m.evolution_history,
            "category":          m.category,
            "tags":              m.tags,
        }
        for m in system.memories.values()
    ]


def load_memories_from_dicts(system, dicts: list[dict]):
    """Reconstruct ``system.memories`` from serialized dicts."""
    from agentic_memory.memory_system import MemoryNote

    system.memories.clear()
    for d in dicts:
        note = MemoryNote(
            content=d["content"],
            id=d["id"],
            keywords=d.get("keywords", []),
            links=d.get("links", []),
            retrieval_count=d.get("retrieval_count", 0),
            timestamp=d.get("timestamp"),
            last_accessed=d.get("last_accessed"),
            context=d.get("context", "General"),
            evolution_history=d.get("evolution_history", []),
            category=d.get("category", "Uncategorized"),
            tags=d.get("tags", []),
        )
        system.memories[note.id] = note
    logger.debug(f"Loaded {len(system.memories)} memories into system")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def build_evidence_map(item: dict, note_registry: dict) -> dict:
    """Identify note IDs containing evidence for *item*."""
    evidence_plan = item.get("evidence_plan", [])
    item_id = item.get("question_id", "unknown")
    evidence_fact_keys = {ev["fact_key"] for ev in evidence_plan}
    required_fact_keys = {
        ev["fact_key"] for ev in evidence_plan if ev.get("required_for_answer")
    }

    evidence_note_ids: set[str] = set()
    answer_note_ids: set[str] = set()
    fact_key_to_note_ids: dict[str, set[str]] = defaultdict(set)

    for note_id, meta in note_registry.items():
        if meta["item_id"] != item_id:
            continue
        fk = meta.get("fact_key")
        if fk and fk in evidence_fact_keys:
            evidence_note_ids.add(note_id)
            fact_key_to_note_ids[fk].add(note_id)
        if meta.get("has_answer") is True:
            answer_note_ids.add(note_id)
            evidence_note_ids.add(note_id)

    return {
        "evidence_note_ids":     evidence_note_ids,
        "answer_note_ids":       answer_note_ids,
        "required_note_ids": {
            nid
            for nid, meta in note_registry.items()
            if meta["item_id"] == item_id
            and meta.get("fact_key") in required_fact_keys
        },
        "fact_key_to_note_ids": dict(fact_key_to_note_ids),
        "num_evidence": len(evidence_note_ids),
        "num_required": len(required_fact_keys),
    }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    search_results: list[dict], evidence_note_ids: set[str], k: int,
) -> dict:
    """Compute Recall@k, Precision@k, MRR, NDCG@k."""
    top_k_ids = [r["id"] for r in search_results[:k]]
    num_evidence = len(evidence_note_ids)

    if num_evidence == 0:
        return {
            "recall": None, "precision": None, "mrr": None, "ndcg": None,
            "hits": 0, "num_evidence": 0, "k": k,
        }

    hits = sum(1 for nid in top_k_ids if nid in evidence_note_ids)
    recall = hits / num_evidence
    precision = hits / k

    mrr = 0.0
    for rank, nid in enumerate(top_k_ids, start=1):
        if nid in evidence_note_ids:
            mrr = 1.0 / rank
            break

    dcg = 0.0
    idcg = 0.0
    for i in range(k):
        rel = 1.0 if i < len(top_k_ids) and top_k_ids[i] in evidence_note_ids else 0.0
        dcg += rel / math.log2(i + 2)
    for i in range(min(num_evidence, k)):
        idcg += 1.0 / math.log2(i + 2)
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall":       round(recall, 4),
        "precision":    round(precision, 4),
        "mrr":          round(mrr, 4),
        "ndcg":         round(ndcg, 4),
        "hits":         hits,
        "num_evidence": num_evidence,
        "k":            k,
    }


# ---------------------------------------------------------------------------
# Optional QA evaluation (LLM-as-judge)
# ---------------------------------------------------------------------------


def _evidence_update_answer_rule(question_type: str | None = None) -> str:
    """Local copy of the shared KU answer rule.

    This module is imported as ``_common`` inside ``evaluation/amem/``, so it
    must not ``from _common import ...`` (that resolves to this file).
    """
    if str(question_type or "").strip().lower() == "knowledge_update":
        return (
            "If this is a knowledge_update question (what changed / was updated), "
            "answer with the full transition as \"from <old> to <new>\". "
            "Do not answer with only the final/current state when both the earlier "
            "and later values appear in the evidence."
        )
    return (
        "If the evidence contains updates or contradictions about a user's "
        "state, preferences, or attributes, the most recent information takes "
        "precedence."
    )


def _call_llm(messages: list, llm_config: dict) -> str:
    """OpenAI-compatible chat completion used by the A-MEM answer step."""
    import httpx
    from openai import OpenAI

    if not hasattr(_call_llm, "clients"):
        _call_llm.clients = {}
    base_url = llm_config["base_url"]
    if base_url not in _call_llm.clients:
        _call_llm.clients[base_url] = OpenAI(
            base_url=base_url,
            api_key=llm_config["api_key"],
            http_client=httpx.Client(proxy=llm_config.get("proxy")),
        )
    extra_params = dict(llm_config.get("extra_params", {}))
    if llm_config.get("thinking_disabled", False):
        extra_body = dict(extra_params.get("extra_body", {}))
        extra_body["thinking"] = {"type": "disabled"}
        extra_params["extra_body"] = extra_body
    response = _call_llm.clients[base_url].chat.completions.create(
        model=llm_config["llm_model"],
        messages=messages,
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        **extra_params,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("LLM returned an empty answer")
    return content


def evaluate_qa(
    item: dict, search_results: list[dict], llm_config: dict,
) -> dict | None:
    """Use an LLM to answer a question based on retrieved context.

    The expected answer is NOT shown to the LLM — it must derive the answer
    from the retrieved context alone.
    """
    question = item.get("question", "")

    context_parts = []
    context_top_k = llm_config.get("context_top_k")
    delivered_results = (
        search_results[: int(context_top_k)] if context_top_k else search_results
    )
    for i, r in enumerate(delivered_results):
        context_parts.append(f"[{i + 1}] {r.get('content', '')}")
    context_str = "\n".join(context_parts)

    question_type = item.get("question_type")
    system_prompt = (
        "You are an evaluation assistant. Your task is to answer questions "
        "about a user based on retrieved conversation history.\n\n"
        "Answer rules:\n"
        "1. Answer using ONLY the retrieved evidence provided. Do not use "
        "outside knowledge.\n"
        "2. Your answer should be as short as possible — typically a single "
        "phrase, number, name, or date.\n"
        "3. Do not include reasoning, explanations, or hedging in your final "
        "answer. Do not describe the multimodal content; extract the "
        "requested fact directly.\n"
        "4. If the retrieved evidence does not contain enough information to "
        "answer the question, respond with exactly: I don't know.\n"
        f"5. {_evidence_update_answer_rule(question_type)}\n"
        "6. Examples of correct answer format:\n"
        '   - Question: "What was the work start time?" → Answer: "8:15 AM"\n'
        '   - Question: "What did the charity race raise awareness for?" → Answer: "mental health"'
    )

    user_prompt = (
        "<RETRIEVED CONVERSATION HISTORY>\n"
        f"{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(1, 4):
        try:
            answer = _call_llm(messages, llm_config).strip()
            # Strip common wrappers the LLM may add
            for prefix in ("Answer: ", "answer: "):
                if answer.startswith(prefix):
                    answer = answer[len(prefix):]
            return {"predicted_answer": answer}
        except Exception as e:
            last_error = e
            logger.warning(
                f"QA eval LLM call failed (attempt {attempt}/3) for "
                f"{item.get('question_id', '?')}: {e}"
            )

    logger.error(
        f"QA eval failed after 3 retries for {item.get('question_id', '?')}: "
        f"{last_error}"
    )
    return {"error": str(last_error)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_results(item_results: list[dict], k_values: list[int]) -> dict:
    """Compute mean / std for each metric, overall and by question_type."""
    overall: dict[str, list[float]] = defaultdict(list)
    by_type: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for r in item_results:
        if "error" in r:
            continue
        qt = r.get("question_type", "unknown")
        for k_label, metrics in r.get("metrics", {}).items():
            for metric_name in ["recall", "precision", "mrr", "ndcg"]:
                val = metrics.get(metric_name)
                if val is not None:
                    key = f"{metric_name}{k_label}"
                    overall[key].append(val)
                    by_type[qt][key].append(val)

    def _summarize(vals):
        if not vals:
            return {"mean": None, "std": None, "count": 0}
        return {
            "mean":  round(float(np.mean(vals)), 4),
            "std":   round(float(np.std(vals)), 4),
            "count": len(vals),
        }

    return {
        "overall": {k: _summarize(v) for k, v in sorted(overall.items())},
        "by_question_type": {
            qt: {k: _summarize(v) for k, v in sorted(m.items())}
            for qt, m in sorted(by_type.items())
        },
        "k_values": k_values,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_results(summary: dict, item_results: list[dict], output_dir: str):
    """Write *summary.json* and *details.jsonl* to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    details_path = os.path.join(output_dir, "details.jsonl")
    with open(details_path, "w") as f:
        for r in item_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Summary  -> {summary_path}")
    logger.info(f"Details  -> {details_path}")


def print_console_summary(summary: dict):
    """Print a compact summary table to the console."""
    k_values = summary.get("k_values", [])
    overall = summary.get("overall", {})

    print("\n" + "=" * 72)
    print("A-MEM + MOSAIC  ---  EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"Personas:   {summary.get('num_personas', '?')} processed, "
        f"{summary.get('num_personas_failed', 0)} failed"
    )
    print(
        f"Items:      {summary.get('successful_items', '?')} OK, "
        f"{summary.get('failed_items', '?')} failed "
        f"(of {summary.get('total_items', '?')} total)"
    )
    print()

    header = f"{'Metric':<14}" + "".join(f"{'@' + str(k):>10}" for k in k_values)
    print(header)
    print("-" * len(header))

    for metric in ["recall", "precision", "mrr", "ndcg"]:
        parts = [f"{metric:<14}"]
        for k in k_values:
            key = f"{metric}@{k}"
            entry = overall.get(key, {})
            mean = entry.get("mean")
            parts.append(f"{mean:>10.4f}" if mean is not None else f"{'N/A':>10}")
        print("".join(parts))

    by_type = summary.get("by_question_type", {})
    if by_type:
        print(f"\n{'─' * 72}")
        print("Recall@5 by question type:")
        for qt, metrics in sorted(by_type.items()):
            entry = metrics.get("recall@5", {})
            mean = entry.get("mean")
            count = entry.get("count", 0)
            print(
                f"  {qt:<30s} {mean:.4f}  (n={count})"
                if mean is not None else
                f"  {qt:<30s} N/A  (n={count})"
            )

    print("=" * 72)
