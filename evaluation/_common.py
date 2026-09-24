#!/usr/bin/env python3
"""Shared utilities for the MMLongMemEval evaluation pipeline.

Used by build_index_*.py and eval_*.py scripts.
No dependency on A-MEM or MemoryOS — pure FAISS + sentence-transformers.

"""

from __future__ import annotations

import json
import os
import sys
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
from loguru import logger

try:
    from .protocol import (
        DeliveryPolicy,
        compute_retrieval_recall as _compute_protocol_retrieval_recall,
        materialize_openai_media_blocks,
        pack_evidence,
        retrieval_metric_aggregate,
    )
except ImportError:  # Direct execution with evaluation/ on sys.path.
    from protocol import (
        DeliveryPolicy,
        compute_retrieval_recall as _compute_protocol_retrieval_recall,
        materialize_openai_media_blocks,
        pack_evidence,
        retrieval_metric_aggregate,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TIMESTAMP_FMT = "%Y/%m/%d (%a) %H:%M"
FALLBACK_DATE_FMT = "%Y/%m/%d"
UNANSWERABLE_PATTERNS = [
    "i don't know",
    "i don't know.",
    "cannot determine",
    "cannot answer",
    "not enough information",
    "insufficient information",
    "unable to determine",
    "unable to answer",
    "unknown",
    "n/a",
    "none",
    "not specified",
    "unanswerable",
]

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> dict:
    """Load JSON config, resolving API keys / base URLs from env vars.

    Supports both ``"api_key": "sk-..."`` (direct value) and
    ``"api_key_env": "OPENAI_API_KEY"`` (env-var lookup).
    Direct value wins. Also supports ``base_url_env`` /
    ``openai_base_url_env``, and fills ``base_url`` from provider URL
    aliases when ``api_key_env`` is DeepSeek / OpenRouter / Yunwu.

    Loads the repo ``.env`` (via :mod:`env_bootstrap`) on first use so
    scripts do not require a manual ``source .env``.
    """
    from env_bootstrap import (
        ensure_env_loaded,
        expand_mosaic_paths,
        resolve_base_url_for_api_key_env,
    )

    ensure_env_loaded()

    with open(config_path) as f:
        config = json.load(f)

    def _resolve_keys(value: Any):
        if isinstance(value, dict):
            env_var = value.get("api_key_env") or value.get("openai_api_key_env")
            if not value.get("api_key") and env_var:
                value["api_key"] = os.getenv(str(env_var), "")

            base_url_env = value.get("base_url_env")
            if base_url_env and os.getenv(str(base_url_env)):
                value["base_url"] = os.getenv(str(base_url_env), "")

            openai_base_url_env = value.get("openai_base_url_env")
            if openai_base_url_env and os.getenv(str(openai_base_url_env)):
                value["openai_base_url"] = os.getenv(str(openai_base_url_env), "")

            # Provider convention: override hardcoded base_url from .env URLs.
            if env_var:
                resolved = resolve_base_url_for_api_key_env(str(env_var))
                if resolved:
                    if "base_url" in value:
                        value["base_url"] = resolved
                    if "openai_base_url" in value:
                        value["openai_base_url"] = resolved

            for child in value.values():
                _resolve_keys(child)
        elif isinstance(value, list):
            for child in value:
                _resolve_keys(child)

    _resolve_keys(config)
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
# Session deduplication & chunking
# ---------------------------------------------------------------------------


def deduplicate_sessions(items: list[dict]) -> list[dict]:
    """Deduplicate sessions across items by session_id, sorted by timestamp."""
    seen: dict[str, dict] = {}
    for item in items:
        for session in item.get("haystack_sessions", []):
            sid = session.get("session_id", "")
            if sid and sid not in seen:
                seen[sid] = session

    def _sort_key(s: dict) -> str:
        return s.get("timestamp", s.get("session_date", ""))

    return sorted(seen.values(), key=_sort_key)


def chunk_messages_textualized(sessions: list[dict]) -> list[dict]:
    """Create text chunks from sessions (B1/B2) — one chunk per session.

    All messages in a session are concatenated into a single text chunk.
    Each message's content (or textualized field for multimodal messages)
    is joined with newlines.

    Returns list of chunk dicts with: chunk_id, content, session_id,
    timestamp, modality, fact_keys, has_answer.
    """
    chunks: list[dict] = []
    for session in sessions:
        session_id = session.get("session_id", "unknown")
        timestamp = session.get("timestamp", session.get("session_date", ""))

        parts: list[str] = []
        fact_keys: list[str] = []
        has_answer = False

        for msg in session.get("messages", []):
            mm = msg.get("multimodal_content")
            if mm and mm.get("textualized"):
                parts.append(f"[{msg.get('role', '?')}] {mm['textualized']}")
            else:
                content = msg.get("content", "")
                if content.strip():
                    parts.append(f"[{msg.get('role', '?')}] {content}")

            fk = msg.get("fact_key")
            if fk:
                fact_keys.append(fk)
            if msg.get("has_answer"):
                has_answer = True

        if not parts:
            continue

        chunks.append({
            "chunk_id":     session_id,
            "content":      "\n".join(parts),
            "session_id":   session_id,
            "timestamp":    timestamp,
            "modality":     "text",
            "fact_key":     fact_keys[0] if len(fact_keys) == 1 else fact_keys,
            "fact_keys":    fact_keys,
            "has_answer":   has_answer,
            "role":         "mixed",
        })

    return chunks


def chunk_messages_multimodal(sessions: list[dict]) -> list[dict]:
    """Create chunks from sessions (Omni-MM) — one session text chunk + one per multimodal piece.

    - Session text chunk: all text-only messages concatenated → 1 embedding.
    - Multimodal chunks: each image/video/audio gets its own embedding
      (text + media file fused via jina).

    Returns list of chunk dicts with: chunk_id, document (dict for encode_document),
    session_id, timestamp, modality, fact_key/fact_keys, has_answer, media_path.
    """
    chunks: list[dict] = []
    for session in sessions:
        session_id = session.get("session_id", "unknown")
        timestamp = session.get("timestamp", session.get("session_date", ""))

        # Collect text messages for the session-level chunk
        text_parts: list[str] = []
        text_fact_keys: list[str] = []
        text_has_answer = False

        for i, msg in enumerate(session.get("messages", [])):
            content = msg.get("content", "")
            mm = msg.get("multimodal_content")

            if mm and mm.get("media_path"):
                # --- Individual multimodal chunk ---
                media_path = mm["media_path"]
                modality = mm.get("modality", "text")

                if not os.path.exists(media_path):
                    logger.debug(f"Media file missing: {media_path}, falling back to text")
                    document = {"text": content}
                    modality = "text"
                    media_path = None
                else:
                    document = {"text": content, modality: media_path}

                chunk_id = f"{session_id}_mm{i}"
                chunks.append({
                    "chunk_id":     chunk_id,
                    "document":     document,
                    "content":      content,
                    "session_id":   session_id,
                    "timestamp":    timestamp,
                    "modality":     modality,
                    "fact_key":     msg.get("fact_key"),
                    "has_answer":   msg.get("has_answer"),
                    "role":         msg.get("role", "unknown"),
                    "media_path":   media_path,
                })
            else:
                # --- Text-only message: accumulate for session chunk ---
                if content.strip():
                    text_parts.append(f"[{msg.get('role', '?')}] {content}")
                fk = msg.get("fact_key")
                if fk:
                    text_fact_keys.append(fk)
                if msg.get("has_answer"):
                    text_has_answer = True

        # Session-level text chunk
        if text_parts:
            session_text = "\n".join(text_parts)
            chunks.append({
                "chunk_id":     session_id,
                "document":     {"text": session_text},
                "content":      session_text,
                "session_id":   session_id,
                "timestamp":    timestamp,
                "modality":     "text",
                "fact_key":     text_fact_keys[0] if len(text_fact_keys) == 1 else text_fact_keys,
                "fact_keys":    text_fact_keys,
                "has_answer":   text_has_answer,
                "role":         "mixed",
                "media_path":   None,
            })

    return chunks


# ---------------------------------------------------------------------------
# Text embedding singleton (B1, B2, B4)
# ---------------------------------------------------------------------------

_shared_text_model: object | None = None
_text_model_lock = threading.Lock()
_text_encode_lock = threading.Lock()


def get_text_embedding_model(model_name: str, device: str = "cpu",
                            model_kwargs: dict | None = None,
                            processor_kwargs: dict | None = None) -> object:
    """Return a thread-safe singleton text embedding model.

    Idempotent — only the first caller loads weights.

    Parameters
    ----------
    model_name:
        HuggingFace model ID (e.g. ``"Qwen/Qwen3-Embedding-0.6B"``).
    device:
        Device string (``"cuda:0"``, ``"cpu"``, etc.).
    model_kwargs:
        Passed to SentenceTransformer (e.g. ``{"attn_implementation": "flash_attention_2"}``).
    processor_kwargs:
        Passed to SentenceTransformer (e.g. ``{"padding_side": "left"}``).
    """
    global _shared_text_model
    if _shared_text_model is not None:
        return _shared_text_model

    with _text_model_lock:
        if _shared_text_model is not None:
            return _shared_text_model

        from sentence_transformers import SentenceTransformer

        # Copy before resolving dtypes so we never mutate the caller's config
        # (mutated torch.dtype values break run_manifest JSON serialization).
        resolved_model_kwargs = dict(model_kwargs) if model_kwargs else None
        if resolved_model_kwargs and "torch_dtype" in resolved_model_kwargs:
            import torch as _torch
            dt = resolved_model_kwargs["torch_dtype"]
            if isinstance(dt, str):
                resolved_model_kwargs["torch_dtype"] = getattr(_torch, dt)

        st_kwargs: dict[str, Any] = {}
        if resolved_model_kwargs:
            st_kwargs["model_kwargs"] = resolved_model_kwargs
        if processor_kwargs:
            st_kwargs["processor_kwargs"] = dict(processor_kwargs)

        try:
            _shared_text_model = SentenceTransformer(
                model_name, device=device, **st_kwargs
            )
        except TypeError as exc:
            # Older sentence-transformers builds reject processor_kwargs.
            if "processor_kwargs" not in st_kwargs:
                raise
            logger.warning(
                "SentenceTransformer rejected processor_kwargs ({}); retrying without it",
                exc,
            )
            st_kwargs.pop("processor_kwargs", None)
            _shared_text_model = SentenceTransformer(
                model_name, device=device, **st_kwargs
            )
        logger.info(f"Loaded text embedding model '{model_name}' on '{device}'")

        dim = _shared_text_model.get_embedding_dimension()
        logger.info(f"  Embedding dimension: {dim}")

        return _shared_text_model


def encode_text_batch(texts: list[str], model_name: str, device: str = "cpu",
                      prompt_name: str | None = None) -> np.ndarray:
    """Encode a batch of texts through the shared text model (thread-safe).

    Parameters
    ----------
    texts:
        List of text strings to encode.
    model_name:
        HuggingFace model ID.
    device:
        Device string.
    prompt_name:
        For instruction-aware models (e.g. Qwen3-Embedding), set to ``"query"``
        when encoding questions to get +1-5% retrieval performance.
        Leave as ``None`` when encoding document chunks.
    """
    model = get_text_embedding_model(model_name, device)
    with _text_encode_lock:
        kwargs = dict(
            batch_size=8,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        embeddings = model.encode(texts, **kwargs)
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# Omni/multimodal embedding singleton (B3, B6) — uses jina-embeddings-v5-omni-small
# ---------------------------------------------------------------------------

_shared_omni_model: object | None = None
_omni_model_lock = threading.Lock()
_omni_encode_lock = threading.Lock()


def get_omni_embed_model(config: dict | None = None) -> object:
    """Return a thread-safe singleton multimodal embedding model.

    Uses ``jinaai/jina-embeddings-v5-omni-small`` via sentence_transformers.
    Supports text, image, video, and audio inputs.

    Returns a SentenceTransformer model instance.
    """
    global _shared_omni_model
    if _shared_omni_model is not None:
        return _shared_omni_model

    with _omni_model_lock:
        if _shared_omni_model is not None:
            return _shared_omni_model

        from sentence_transformers import SentenceTransformer

        cfg = config or {}
        model_name = cfg.get("model_name", "jinaai/jina-embeddings-v5-omni-small")
        model_kwargs = {"default_task": "retrieval"}

        _shared_omni_model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs=model_kwargs,
            device=cfg.get("device", "cuda:0"),
        )

        dim = _shared_omni_model.get_embedding_dimension()
        logger.info(f"Loaded jina omni model '{model_name}' (dim={dim})")

        return _shared_omni_model


def _to_jina_document(chunk_doc: dict) -> tuple:
    """Convert our internal chunk document dict to a jina-format tuple.

    Always returns a tuple — single-element for text-only, multi-element
    for text + media.  Tuples bypass jina's URL sniffing on bare strings
    (conversation text may contain ``http://`` links that would otherwise
    be misinterpreted as remote media files).

    Images are opened with PIL to bypass content-sniffing (which can
    falsely detect SVG in AI-generated PNG metadata).
    """
    text = chunk_doc.get("text", "")
    # Prevent jina from treating bare-URL messages as remote media files
    # (12 benchmark messages start with https:// as their full content).
    if text.strip().startswith(("http://", "https://")):
        text = "URL: " + text
    media_parts = []
    for modality in ("image", "video", "audio"):
        path = chunk_doc.get(modality)
        if not (path and os.path.exists(path)):
            continue
        if modality == "image":
            from PIL import Image
            media_parts.append(Image.open(path).convert("RGB"))
        else:
            media_parts.append(path)

    return (text, *media_parts)


def encode_omni_documents(documents: list[dict], config: dict | None = None) -> np.ndarray:
    """Encode document chunks via jina model.encode_document() (thread-safe).

    All docs are passed as tuples to bypass URL sniffing.
    """
    model = get_omni_embed_model(config)
    jina_docs = [_to_jina_document(d) for d in documents]
    with _omni_encode_lock:
        embeddings = model.encode_document(jina_docs)
    return np.asarray(embeddings, dtype=np.float32)


def encode_omni_queries(queries: list[str], config: dict | None = None) -> np.ndarray:
    """Encode query texts via jina model.encode_query() (thread-safe)."""
    model = get_omni_embed_model(config)
    with _omni_encode_lock:
        embeddings = model.encode_query(queries)
    return np.asarray(embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAISS index helpers
# ---------------------------------------------------------------------------


def create_faiss_index(dimension: int, index_type: str = "flat") -> object:
    """Create a FAISS index for inner-product search (cosine with normalized vectors).

    Parameters
    ----------
    dimension:
        Embedding vector dimension.
    index_type:
        ``"flat"`` for IndexFlatIP (exact, good for <100K vectors).
        ``"ivf"`` for IndexIVFFlat (approximate, larger scale).
    """
    import faiss

    if index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dimension)
        nlist = min(100, max(4, dimension // 16))
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = 10
    else:
        index = faiss.IndexFlatIP(dimension)

    return index


def build_faiss_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    index_path: str,
    meta_path: str,
    index_type: str = "flat",
) -> object:
    """Create, populate, and persist a FAISS index.

    Parameters
    ----------
    embeddings:
        Float32 array of shape (N, D).  Must be L2-normalized for cosine similarity.
    metadata:
        List of N metadata dicts (one per embedding).
    index_path:
        Path for ``faiss.write_index()``.
    meta_path:
        Path for metadata JSON.
    index_type:
        Passed to ``create_faiss_index()``.

    Returns the populated FAISS index.
    """
    import faiss

    N, D = embeddings.shape
    index = create_faiss_index(D, index_type)

    if index_type == "ivf" and N > 0:
        index.train(embeddings)

    index.add(embeddings)
    faiss.write_index(index, index_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, ensure_ascii=False)

    logger.debug(f"Built FAISS index: {N} vectors x {D}d -> {index_path}")
    return index


def load_faiss_index(index_path: str, meta_path: str) -> tuple[object, list[dict]]:
    """Load a persisted FAISS index and its metadata."""
    import faiss

    index = faiss.read_index(index_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return index, metadata


def search_faiss_index(
    index: object,
    query_vector: np.ndarray,
    metadata: list[dict],
    k: int = 20,
) -> list[dict]:
    """Search FAISS index and attach metadata to results.

    Parameters
    ----------
    index:
        FAISS index.
    query_vector:
        Shape (1, D) float32 query vector.
    metadata:
        List of metadata dicts (same order as index).
    k:
        Number of results to return.

    Returns list of result dicts with score and metadata.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    k_actual = min(k, index.ntotal)
    if k_actual == 0:
        return []

    distances, indices = index.search(query_vector, k_actual)

    results = []
    for pre_filter_rank, (dist, idx) in enumerate(
        zip(distances[0], indices[0]), start=1
    ):
        if idx < 0 or idx >= len(metadata):
            continue
        result = dict(metadata[idx])
        result["score"] = float(dist)
        result["pre_filter_rank"] = pre_filter_rank
        result["rank"] = pre_filter_rank
        results.append(result)

    return results


def search_faiss_index_filtered(
    index: object,
    query_vector: np.ndarray,
    metadata: list[dict],
    *,
    k: int = 20,
    candidate_k: int | None = None,
    eligibility_fn=None,
    max_candidate_k: int | None = None,
) -> list[dict]:
    """Retrieve a full eligible top-k with iterative candidate expansion.

    Current final-snapshot runs do not need an eligibility filter.  This helper
    prevents future timestamp/score protocols from asking FAISS for exactly
    ``k`` and then silently returning fewer than ``k`` after filtering.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return []
    eligibility_fn = eligibility_fn or (lambda unit: True)
    total = int(getattr(index, "ntotal", len(metadata)))
    ceiling = min(total, max_candidate_k or total)
    pool = max(k, candidate_k or k)
    pool = min(pool, ceiling)
    eligible: list[dict] = []
    while pool > 0:
        candidates = search_faiss_index(index, query_vector, metadata, k=pool)
        eligible = []
        for candidate in candidates:
            if eligibility_fn(candidate):
                selected = dict(candidate)
                selected["eligible_rank"] = len(eligible) + 1
                selected["rank"] = len(eligible) + 1
                eligible.append(selected)
                if len(eligible) == k:
                    return eligible
        if pool >= ceiling:
            break
        pool = min(ceiling, max(pool + 1, pool * 2))
    return eligible[:k]


# ---------------------------------------------------------------------------
# Answer matching
# ---------------------------------------------------------------------------


def is_knowledge_update_question(question_type: str | None) -> bool:
    """True when the item is a knowledge_update question."""
    return str(question_type or "").strip().lower() == "knowledge_update"


def evidence_update_answer_rule(question_type: str | None = None) -> str:
    """Instruction for conflicting/updated evidence, specialized for KU."""
    if is_knowledge_update_question(question_type):
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


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.strip().lower()
    # Remove common prefixes
    for prefix in ["answer:", "the answer is:", "response:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Collapse whitespace
    return " ".join(text.split())


def match_answer(
    predicted: str,
    expected: str,
    answer_aliases: list[str] | None = None,
    answerability: str = "answerable",
) -> dict:
    """Compare predicted answer with ground truth.

    For answerable questions: exact match + alias match.
    For unanswerable questions: check against refusal patterns.
    """
    norm_pred = _normalize(predicted)
    norm_expected = _normalize(expected)
    norm_aliases = [_normalize(a) for a in (answer_aliases or [])]

    if answerability == "unanswerable":
        unanswerable_correct = any(
            pattern in norm_pred for pattern in UNANSWERABLE_PATTERNS
        )
        return {
            "exact_match": False,
            "alias_match": unanswerable_correct,
            "unanswerable_correct": unanswerable_correct,
            "normalized_predicted": norm_pred,
            "normalized_ground_truth": norm_expected,
        }

    exact = norm_pred == norm_expected
    alias = exact or any(norm_pred == a for a in norm_aliases)

    return {
        "exact_match": exact,
        "alias_match": alias,
        "unanswerable_correct": False,
        "normalized_predicted": norm_pred,
        "normalized_ground_truth": norm_expected,
    }


# ---------------------------------------------------------------------------
# Retrieval recall
# ---------------------------------------------------------------------------


def compute_retrieval_recall(
    retrieved: list[dict],
    evidence_plan: list[dict],
    k_values: list[int] = (1, 3, 5, 10, 20),
) -> dict:
    """Compute rank-sensitive fact and complete-chain retrieval metrics."""
    return _compute_protocol_retrieval_recall(retrieved, evidence_plan, k_values)


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def aggregate_results(item_results: list[dict]) -> dict:
    """Compute accuracy metrics from per-question results.

    Each result has: question_type, answerability, evaluation dict (from match_answer).
    """
    total = len(item_results)
    if total == 0:
        return {"total_questions": 0}

    answerable = [r for r in item_results if r.get("answerability") == "answerable"]
    unanswerable = [r for r in item_results if r.get("answerability") == "unanswerable"]

    correct_answerable = sum(
        1 for r in answerable
        if r.get("evaluation", {}).get("alias_match", False)
    )
    correct_unanswerable = sum(
        1 for r in unanswerable
        if r.get("evaluation", {}).get("unanswerable_correct", False)
    )

    # Per question_type
    by_type: dict[str, list] = defaultdict(list)
    for r in item_results:
        by_type[r.get("question_type", "unknown")].append(r)

    per_type = {}
    for qtype, type_items in sorted(by_type.items()):
        ta = [r for r in type_items if r.get("answerability") == "answerable"]
        tu = [r for r in type_items if r.get("answerability") == "unanswerable"]
        tc = (
            sum(1 for r in ta if r.get("evaluation", {}).get("alias_match", False))
            + sum(1 for r in tu if r.get("evaluation", {}).get("unanswerable_correct", False))
        )
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
        "retrieval": retrieval_metric_aggregate(item_results),
    }


def partition_terminal_results(
    rows: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split terminal rows by observable API status."""
    successful = [row for row in rows if row.get("api_status") == "ok"]
    failed = [row for row in rows if row.get("api_status") != "ok"]
    return successful, failed


# ---------------------------------------------------------------------------
# Resume / retry helpers
# ---------------------------------------------------------------------------


def load_failed_entries(details_path: str) -> tuple[list[dict], dict[str, set[str]]]:
    """Load details.jsonl and find failed entries for --resume.

    Returns (all_entries, persona_to_failed_qids).
    A failed entry has ``error`` field or ``predicted_answer`` starting
    with ``"ERROR:"``.
    """
    import json as _json

    all_entries: list[dict] = []
    persona_failed: dict[str, set[str]] = {}

    with open(details_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            entry = _json.loads(line)
            entry["_line_idx"] = i
            all_entries.append(entry)

            # Check if failed
            is_error = (
                "error" in entry
                or (entry.get("predicted_answer") or "").startswith("ERROR:")
                or entry.get("api_status") in {"timeout", "error", "empty"}
            )
            if is_error:
                puid = entry.get("persona_uuid", "unknown")
                qid = entry.get("question_id", "")
                if puid not in persona_failed:
                    persona_failed[puid] = set()
                persona_failed[puid].add(qid)

    return all_entries, persona_failed


# ---------------------------------------------------------------------------
# Output helpers
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


# ---------------------------------------------------------------------------
# Shared persona-level build (used by B1 and B2)
# ---------------------------------------------------------------------------


def build_text_index_for_persona(
    filepath: str,
    index_dir: str,
    embedding_model_name: str,
    embedding_device: str = "cpu",
    batch_size: int = 256,
    index_type: str = "flat",
    pbar=None,
) -> dict | None:
    """Build a text FAISS index for a single persona (B1/B2).

    1. Load persona file
    2. Deduplicate sessions
    3. Chunk messages (textualized)
    4. Embed chunks with text model
    5. Build + persist FAISS index
    6. Save items.json, persona.json

    Returns a manifest entry dict, or ``None`` on failure.
    """
    try:
        persona, items = load_persona_file(filepath)
    except Exception:
        logger.exception(f"Failed to load {os.path.basename(filepath)}")
        return None

    persona_uuid = persona.get("uuid", os.path.basename(filepath))
    persona_name = persona.get("name", "unknown")
    out_dir = os.path.join(index_dir, persona_uuid)

    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    # Per-persona log
    sink_id = logger.add(
        os.path.join(out_dir, "build.log"), level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    try:
        import time
        t0 = time.time()

        sessions = deduplicate_sessions(items)
        chunks = chunk_messages_textualized(sessions)
        logger.info(f"[{persona_name}] {len(sessions)} sessions, {len(chunks)} chunks")

        if not chunks:
            logger.warning(f"[{persona_name}] No chunks — skipping")
            return None

        # Embed chunks in batches
        all_texts = [c["content"] for c in chunks]
        all_embeddings = []
        for i in range(0, len(all_texts), batch_size):
            batch_texts = all_texts[i:i + batch_size]
            batch_embs = encode_text_batch(batch_texts, embedding_model_name, embedding_device)
            all_embeddings.append(batch_embs)
            if pbar is not None:
                pbar.update(len(batch_texts))

        embeddings = np.concatenate(all_embeddings, axis=0)
        logger.debug(f"[{persona_name}] Encoded {len(chunks)} chunks -> {embeddings.shape}")

        # Build FAISS index
        index_path = os.path.join(out_dir, "index.faiss")
        meta_path = os.path.join(out_dir, "metadata.json")
        build_faiss_index(embeddings, chunks, index_path, meta_path, index_type)

        # Save items + persona for downstream eval
        with open(os.path.join(out_dir, "items.json"), "w") as f:
            json.dump(items, f, ensure_ascii=False)
        with open(os.path.join(out_dir, "persona.json"), "w") as f:
            json.dump(persona, f, ensure_ascii=False)

        elapsed = time.time() - t0
        logger.info(
            f"[{persona_name}] {len(chunks)} chunks, "
            f"{len(sessions)} sessions, {len(items)} items "
            f"in {elapsed:.1f}s -> {out_dir}"
        )

        return {
            "persona_uuid":  persona_uuid,
            "persona_name":  persona_name,
            "file":          os.path.basename(filepath),
            "num_chunks":    len(chunks),
            "num_sessions":  len(sessions),
            "num_items":     len(items),
            "path":          out_dir,
            "elapsed":       round(elapsed, 1),
        }
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Shared persona-level build (multimodal, used by B3)
# ---------------------------------------------------------------------------


def build_multimodal_index_for_persona(
    filepath: str,
    index_dir: str,
    omni_config: dict,
    batch_size: int = 8,
    index_type: str = "flat",
    pbar=None,
) -> dict | None:
    """Build a multimodal FAISS index for a single persona (Omni-MM).

    Uses jina-embeddings-v5-omni-small via sentence_transformers.
    Text-only and multimodal chunks are both embedded via encode_document().

    Returns a manifest entry dict, or ``None`` on failure.
    """
    try:
        persona, items = load_persona_file(filepath)
    except Exception:
        logger.exception(f"Failed to load {os.path.basename(filepath)}")
        return None

    persona_uuid = persona.get("uuid", os.path.basename(filepath))
    persona_name = persona.get("name", "unknown")
    out_dir = os.path.join(index_dir, persona_uuid)

    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    sink_id = logger.add(
        os.path.join(out_dir, "build.log"), level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    try:
        import time
        t0 = time.time()

        sessions = deduplicate_sessions(items)
        chunks = chunk_messages_multimodal(sessions)
        n_text = sum(1 for c in chunks if c["modality"] == "text")
        n_mm = sum(1 for c in chunks if c["modality"] != "text")
        logger.info(f"[{persona_name}] {len(sessions)} sessions, {len(chunks)} chunks ({n_text} text, {n_mm} mm)")

        if not chunks:
            logger.warning(f"[{persona_name}] No chunks — skipping")
            return None

        all_embeddings = []
        documents = [c["document"] for c in chunks]

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embs = encode_omni_documents(batch, omni_config)
            all_embeddings.append(embs)
            if pbar is not None:
                pbar.update(len(batch))

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)

        # Build FAISS index
        index_path = os.path.join(out_dir, "index.faiss")
        meta_path = os.path.join(out_dir, "metadata.json")
        build_faiss_index(embeddings, chunks, index_path, meta_path, index_type)

        # Save items + persona
        with open(os.path.join(out_dir, "items.json"), "w") as f:
            json.dump(items, f, ensure_ascii=False)
        with open(os.path.join(out_dir, "persona.json"), "w") as f:
            json.dump(persona, f, ensure_ascii=False)

        elapsed = time.time() - t0
        logger.info(
            f"[{persona_name}] {len(chunks)} chunks "
            f"({n_text} text, {n_mm} multimodal), "
            f"{len(sessions)} sessions, {len(items)} items "
            f"in {elapsed:.1f}s -> {out_dir}"
        )

        return {
            "persona_uuid":  persona_uuid,
            "persona_name":  persona_name,
            "file":          os.path.basename(filepath),
            "num_chunks":    len(chunks),
            "num_text":      n_text,
            "num_multimodal": n_mm,
            "num_sessions":  len(sessions),
            "num_items":     len(items),
            "path":          out_dir,
            "elapsed":       round(elapsed, 1),
        }
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Query rewriting (keyword extraction)
# ---------------------------------------------------------------------------


def rewrite_query(question: str, reasoner_config: dict) -> str:
    """Extract keyword search terms from a verbose question.

    Uses the same LLM configured in ``reasoner_config``.  Returns a short,
    keyword-dense query string that embeds closer to evidence chunks than
    the original question does.

    Falls back to the original question if the LLM call fails.
    """
    system_prompt = (
        "You are a query rewriter for a conversation memory search engine. "
        "Given a question, extract the most important KEYWORDS for searching "
        "the conversation history. Return ONLY comma-separated keywords — "
        "no explanations, no full sentences.\n\n"
        "Focus on:\n"
        "- Entity names (people, objects, places)\n"
        "- Key attributes (colors, materials, times)\n"
        "- Actions or events mentioned\n"
        "- Any sensory details (happy tone, photo, voice message)\n\n"
        "Examples:\n"
        'Q: "What was the work start time of the ship\'s telegraph I sounded happy about?"\n'
        "Keywords: ship telegraph, brass-and-glass, happy, work start time, clock\n\n"
        'Q: "In the photo of my road atlas, what state map is tucked in the spiral binding?"\n'
        "Keywords: road atlas photo, spiral binding, state map, tucked in\n\n"
        "Return ONLY the keywords."
    )

    try:
        client = create_openai_client(reasoner_config)
        kwargs: dict = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Q: {question}\nKeywords:"},
            ],
            "temperature": 0.0,
            "max_tokens": 100,
        }
        _apply_extra_body(kwargs, reasoner_config)

        raw = call_llm_with_retry(
            client,
            kwargs,
            max_retries=int(reasoner_config.get("max_retries", 2)),
            backoff=float(reasoner_config.get("retry_backoff", 2.0)),
        )
        if not raw:
            logger.debug(
                "[Query Rewrite] empty response, using original question"
            )
            return question
        keywords = raw.strip()
        # Remove common prefixes the model might add
        for prefix in ("keywords:", "keywords: ", "Keywords: "):
            if keywords.lower().startswith(prefix.lower()):
                keywords = keywords[len(prefix):]
        logger.debug(f"[Query Rewrite] \"{question[:80]}...\" -> \"{keywords}\"")
        return keywords if keywords else question
    except Exception as e:
        logger.debug(f"[Query Rewrite] failed: {e}, using original question")
        return question


# ---------------------------------------------------------------------------
# Extra body helper — model-specific API parameters
# ---------------------------------------------------------------------------


def _apply_extra_body(kwargs: dict, config: dict):
    """Merge model-specific ``extra_body`` from config into API kwargs.

    Supports two modes:

    1. **Explicit** — set ``extra_body`` in config to any dict:
       ``{"extra_body": {"key": "value"}}``

    2. **Legacy shortcut** — ``thinking_disabled: true`` is equivalent to
       ``{"extra_body": {"thinking": {"type": "disabled"}}}`` (DeepSeek).

    The explicit ``extra_body`` takes precedence and is merged on top of the
    legacy shortcut.
    """
    extra = {}
    if config.get("thinking_disabled"):
        extra["thinking"] = {"type": "disabled"}
    if config.get("extra_body"):
        extra.update(config["extra_body"])
    if extra:
        kwargs["extra_body"] = extra


# ---------------------------------------------------------------------------
# Robust JSON parsing
# ---------------------------------------------------------------------------


def parse_json_robust(text: str) -> dict | list:
    """Parse JSON, handling markdown code fences and minor malformations.

    1. Strips ```json ... ``` fences that LLMs often wrap around JSON.
    2. Tries ``json.loads()``.
    3. Falls back to ``json_repair.loads()`` for malformed JSON.
    """
    import json
    import re

    text = text.strip()

    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    m = re.match(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()

    # Try standard JSON first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to json_repair for malformed JSON
    try:
        import json_repair
        return json_repair.loads(text)
    except ImportError:
        logger.warning("json_repair not installed — cannot repair malformed JSON")
        raise
    except Exception:
        raise


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------


def create_openai_client(config: dict):
    """Create an OpenAI-compatible client with explicit configurable timeouts."""
    from openai import OpenAI

    timeout_config = config.get("timeout", config.get("timeouts", 120.0))
    timeout: Any
    if isinstance(timeout_config, dict):
        import httpx

        default = float(timeout_config.get("default", 120.0))
        timeout = httpx.Timeout(
            default,
            connect=float(timeout_config.get("connect", default)),
            read=float(timeout_config.get("read", default)),
            write=float(timeout_config.get("write", default)),
            pool=float(timeout_config.get("pool", default)),
        )
    else:
        timeout = float(timeout_config)
    return OpenAI(
        timeout=timeout,
        api_key=config.get("api_key") or "sk-placeholder",
        base_url=config.get("base_url"),
    )


def call_llm_with_retry(
    client,
    kwargs: dict,
    max_retries: int = 2,
    backoff: float = 2.0,
) -> str:
    """Call OpenAI chat completions with retry on transient failures.

    Returns the stripped text content.  Retries on ``None`` content, empty
    choices, and most network errors.
    """
    import time as _time
    import openai as _openai

    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            choices = getattr(response, "choices", None)
            if choices and len(choices) > 0:
                content = choices[0].message.content
                if content is not None:
                    return content.strip()
                last_error = "API returned None content"
            else:
                last_error = f"API returned no choices (response: {str(response)[:200]})"
        except (_openai.APIError, _openai.APIConnectionError,
                 _openai.RateLimitError, _openai.APITimeoutError) as e:
            last_error = str(e)
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries - 1:
            wait = backoff * (2 ** attempt)
            logger.debug(f"[Retry] attempt {attempt + 1}/{max_retries}: {last_error} — waiting {wait:.1f}s")
            _time.sleep(wait)

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# ReAct agent (iterative retrieval)
# ---------------------------------------------------------------------------


def _format_search_results(results: list[dict], max_results: int = 10) -> str:
    """Format FAISS search results as text for the ReAct agent to consume."""
    if not results:
        return "(no results found)"

    parts = []
    for i, r in enumerate(results[:max_results]):
        parts.append(
            f"[{i + 1}] score={r.get('score', 0):.3f} | "
            f"{r.get('timestamp', '')} | {r.get('modality', 'text')}\n"
            f"    {r.get('content', '')}"
        )
    return "\n\n".join(parts)


def _search_log_add(log: list[dict], query: str, results: list[dict]):
    """Append a search entry to the trace log."""
    log.append({
        "action": "search",
        "query": query,
        "num_results": len(results),
        "top_score": round(results[0]["score"], 4) if results else 0.0,
    })


def _format_answer(
    raw_answer: str,
    question: str,
    reasoner_config: dict,
    client,
) -> str:
    """Extract a short answer from a verbose ReAct response.

    If the answer is already a refusal, return it unchanged.  Otherwise
    call the LLM with a brief extraction prompt to get just the answer.
    """
    # Already a refusal — don't reformat
    norm = raw_answer.lower().strip()
    if any(p in norm for p in ("i don't know", "i don't have", "cannot determine",
                                  "not enough information", "insufficient information")):
        return raw_answer

    # Already short enough
    if len(raw_answer) <= 20:
        return raw_answer

    kwargs = {
        "model": reasoner_config.get("llm_model", "gpt-4o"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an answer extraction assistant. Your task is to "
                    "extract the final answer from the text below.\n\n"
                    "Rules:\n"
                    "1. Your answer should be as short as possible — typically "
                    "a single phrase, number, name, or date. This includes "
                    '"from X to Y" for change/update answers.\n'
                    "2. Do not include reasoning, explanations, or hedging. "
                    "Extract the requested fact directly.\n"
                    "3. If the text indicates the answer is unknown or not "
                    'found, respond with exactly: I don\'t know.\n'
                    "4. Examples of correct answer format:\n"
                    '   - Text: "...the time was 8:15 AM..." → "8:15 AM"\n'
                    '   - Text: "...from $12.00 to $85.00..." → "from $12.00 to $85.00"\n'
                    '   - Text: "...I don\'t have information..." → "I don\'t know"'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Text: {raw_answer}\n\n"
                    f"Extracted answer:"
                ),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 50,
    }
    _apply_extra_body(kwargs, reasoner_config)

    try:
        return call_llm_with_retry(client, kwargs)
    except Exception:
        return raw_answer  # fall back to original


def _react_agent_legacy(
    question: str,
    search_fn,
    reasoner_config: dict,
    max_iterations: int = 5,
    reject_duplicate_queries: bool = True,
) -> tuple[str, list[dict]]:
    """Run a ReAct loop: LLM decides when to search and when to answer.

    Uses OpenAI function calling — the LLM is given a ``search_memory`` tool.

    Parameters
    ----------
    question:
        The question to answer.
    search_fn:
        A callable ``search_fn(query: str) -> str`` that returns formatted
        search results.
    reasoner_config:
        Dict with ``llm_model``, ``base_url``, ``api_key``, ``temperature``,
        ``max_tokens``, ``thinking_disabled``.
    max_iterations:
        Maximum number of search iterations (default 5).
    reject_duplicate_queries:
        If True, detect and reject repeated queries (default True).

    Returns
    -------
    (answer, trace):
        *answer* — the final predicted answer string.
        *trace* — list of dicts recording each iteration (query, num_results,
        top_score) and the final answer.
    """
    client = create_openai_client(reasoner_config)

    system_prompt = (
        "You are a memory retrieval agent. A user asks a question about their "
        "conversation history. You have a search_memory tool to find relevant "
        "information.\n\n"
        "Rules:\n"
        "- If the question can be answered without searching, answer immediately.\n"
        "- If you need more context, search with a specific, keyword-rich query.\n"
        '- Search using short queries (2-5 words), not full sentences.\n'
        "- After 2-3 searches, synthesize an answer from what you've found.\n"
        "- Answer using ONLY the retrieved evidence. Do not use outside knowledge.\n"
        "- Your answer should be as short as possible — a single phrase, number, name, or date.\n"
        "- Do not include reasoning, explanations, or hedging.\n"
        f"- {evidence_update_answer_rule()}\n"
        "- Keep searching until you have enough evidence to answer. Do not "
        "give up early — try different query angles if your first search "
        "doesn't find what you need.\n"
        "- The question may require multiple pieces of evidence (multi-hop). "
        "If one search doesn't find a specific detail, search for a "
        "DIFFERENT piece of evidence instead — you can come back to the "
        "missing detail later. Don't get stuck repeating the same angle.\n"
        "- Final answer: ONLY the requested fact/entity. No markdown. One line.\n"
        "- For knowledge_update questions (what changed / was updated), answer "
        "with the full transition as \"from <old> to <new>\", not only the "
        "final state.\n"
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search the user's conversation history for information "
                "relevant to the question. Returns ranked text excerpts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Short keyword query to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    }]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    trace: list[dict] = []
    seen_queries: set[str] = set()

    for iteration in range(max_iterations):
        kwargs: dict = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": messages,
            "tools": tools,
            "temperature": reasoner_config.get("temperature", 0.0),
            "max_tokens": reasoner_config.get("max_tokens", 1024),
        }
        _apply_extra_body(kwargs, reasoner_config)

        # Retry loop — ReAct needs the raw message for tool_calls
        _msg = None
        _max_retries = int(reasoner_config.get("max_retries", 2))
        _last_exc = None
        for _retry in range(_max_retries):
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as _exc:
                _last_exc = _exc
                logger.debug(
                    f"[ReAct Iter {iteration + 1}] API error "
                    f"(retry {_retry + 1}/{_max_retries}): {_exc}"
                )
                import time as _time2
                _time2.sleep(
                    float(reasoner_config.get("retry_backoff", 2.0))
                    * (2 ** _retry)
                )
                continue
            if getattr(response, "choices", None) and response.choices:
                _msg = response.choices[0].message
                break
            logger.debug(
                f"[ReAct Iter {iteration + 1}] no choices "
                f"(retry {_retry + 1}/{_max_retries})"
            )
            import time as _time2
            _time2.sleep(
                float(reasoner_config.get("retry_backoff", 2.0))
                * (2 ** _retry)
            )
        if _msg is None:
            raise RuntimeError(
                f"ReAct iteration {iteration + 1} exhausted retries: {_last_exc}"
            )
        msg = _msg

        logger.debug(
            f"[ReAct Iter {iteration + 1}] "
            f"tool_calls={bool(msg.tool_calls)}\n"
            f"content={msg.content or ''}"
        )

        # If the model wants to search
        if msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.function.name == "search_memory":
                    args = parse_json_robust(tc.function.arguments)
                    query = args.get("query", "")

                    # Duplicate query rejection
                    if reject_duplicate_queries:
                        q_norm = query.lower().strip()
                        if q_norm in seen_queries:
                            logger.debug(f"[ReAct Search] DUPLICATE: '{query}' — rejected")
                            _search_log_add(trace, query + " (dup)", [])
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tc.id, "type": "function",
                                    "function": {"name": "search_memory",
                                                 "arguments": tc.function.arguments},
                                }],
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": (
                                    f'You already searched for "{query}". '
                                    f"Try a different query or answer based "
                                    f"on what you have."
                                ),
                            })
                            continue
                        seen_queries.add(q_norm)

                    # Execute search
                    results_text = search_fn(query)

                    logger.debug(
                        f"[ReAct Search] query={query}\n"
                        f"results=\n{results_text}"
                    )

                    # Record trace
                    _search_log_add(trace, query, [])

                    # Append assistant tool call + tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": "search_memory",
                                    "arguments": tc.function.arguments,
                                },
                            }
                        ],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": results_text,
                    })
            continue

        # Model returned a final text answer
        answer = (msg.content or "").strip()
        trace.append({"action": "answer", "predicted": answer})
        formatted = _format_answer(answer, question, reasoner_config, client)
        trace.append({"action": "format", "before": answer[:200], "after": formatted})
        return formatted, trace

    # Max iterations reached — ask for final answer
    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of searches. "
            "Based on all the information you've found, give your best answer. "
            'If you still cannot answer, say "I don\'t know."'
        ),
    })

    kwargs = {
        "model": reasoner_config.get("llm_model", "gpt-4o"),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }
    _apply_extra_body(kwargs, reasoner_config)

    answer = call_llm_with_retry(
        client,
        kwargs,
        max_retries=int(reasoner_config.get("max_retries", 2)),
        backoff=float(reasoner_config.get("retry_backoff", 2.0)),
    )
    trace.append({"action": "answer_forced", "predicted": answer})
    formatted = _format_answer(answer, question, reasoner_config, client)
    trace.append({"action": "format", "before": answer[:200], "after": formatted})
    return formatted, trace


class ReActAgentError(RuntimeError):
    """Terminal ReAct failure carrying the observable partial execution."""

    def __init__(
        self,
        message: str,
        *,
        trace: list[dict] | None = None,
        retrieved: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.trace = list(trace or [])
        self.retrieved = list(retrieved or [])


def _react_chat_completion(
    client,
    kwargs: dict,
    *,
    iteration: int,
    max_retries: int,
    retry_backoff: float,
    require_content: bool = False,
) -> tuple[Any | None, dict, Exception | None]:
    """Call one observable ReAct iteration and retain every retry attempt."""
    import time as _time

    attempts: list[dict] = []
    call_started = _time.monotonic()
    message = None
    finish_reason = None
    last_error: Exception | None = None
    for attempt in range(1, max(1, max_retries) + 1):
        started = _time.monotonic()
        try:
            response = client.chat.completions.create(**kwargs)
            choices = getattr(response, "choices", None) or []
            if not choices:
                raise RuntimeError("API returned no choices")
            candidate = choices[0].message
            content = str(getattr(candidate, "content", "") or "").strip()
            if require_content and not content:
                raise RuntimeError("API returned empty content")
            message = candidate
            finish_reason = getattr(choices[0], "finish_reason", None)
            attempts.append({
                "attempt": attempt,
                "status": "ok",
                "latency_seconds": round(_time.monotonic() - started, 3),
            })
            break
        except Exception as error:
            last_error = error
            attempts.append({
                "attempt": attempt,
                "status": "error",
                "exception_class": type(error).__name__,
                "error": str(error),
                "latency_seconds": round(_time.monotonic() - started, 3),
            })
            if attempt < max(1, max_retries):
                _time.sleep(retry_backoff * (2 ** (attempt - 1)))
    return message, {
        "action": "api_call",
        "iteration": iteration,
        "status": "ok" if message is not None else "error",
        "attempts": attempts,
        "latency_seconds": round(_time.monotonic() - call_started, 3),
        "finish_reason": finish_reason,
        "parsed_action": None,
    }, last_error


def react_agent(
    question: str,
    search_fn,
    reasoner_config: dict,
    max_iterations: int = 5,
    reject_duplicate_queries: bool = True,
) -> tuple[str, list[dict], list[dict]]:
    """Run observable function-calling ReAct over actual tool results."""
    client = create_openai_client(reasoner_config)
    system_prompt = (
        "You are a memory retrieval agent. Answer a question about the user's "
        "conversation history using only evidence returned by search_memory. "
        "Issue short, specific search queries and search for every required "
        "hop before answering. Return only the requested fact. If the retrieved "
        "evidence is insufficient, answer exactly: I don't know. For "
        "knowledge_update questions (what changed / was updated), answer with "
        'the full transition as "from <old> to <new>", not only the final state.'
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": "Search the user's conversation memory and return ranked excerpts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]
    trace: list[dict] = []
    seen_queries: set[str] = set()
    retrieved_by_id: dict[str, dict] = {}
    max_retries = max(1, int(reasoner_config.get("max_retries", 2)))
    retry_backoff = float(reasoner_config.get("retry_backoff", 2.0))

    for iteration in range(1, max_iterations + 1):
        kwargs: dict = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": messages,
            "tools": tools,
            "temperature": reasoner_config.get("temperature", 0.0),
            "max_tokens": reasoner_config.get("max_tokens", 1024),
        }
        _apply_extra_body(kwargs, reasoner_config)
        message, call_trace, last_error = _react_chat_completion(
            client,
            kwargs,
            iteration=iteration,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
        )
        trace.append(call_trace)
        if message is None:
            raise ReActAgentError(
                f"ReAct iteration {iteration} exhausted retries: {last_error}",
                trace=trace,
                retrieved=list(retrieved_by_id.values()),
            )

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            call_trace["parsed_action"] = "search_memory"
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", None),
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ],
            })
            for tool_call in tool_calls:
                if tool_call.function.name != "search_memory":
                    error = RuntimeError(
                        f"Unexpected ReAct tool: {tool_call.function.name}"
                    )
                    trace.append({
                        "action": "malformed_tool_call",
                        "iteration": iteration,
                        "status": "error",
                        "error": str(error),
                    })
                    raise ReActAgentError(
                        str(error),
                        trace=trace,
                        retrieved=list(retrieved_by_id.values()),
                    ) from error
                try:
                    arguments = parse_json_robust(tool_call.function.arguments)
                    query = str(arguments.get("query", "")).strip()
                    if not query:
                        raise ValueError("empty search query")
                except Exception as error:
                    trace.append({
                        "action": "malformed_tool_call",
                        "iteration": iteration,
                        "status": "error",
                        "exception_class": type(error).__name__,
                        "error": str(error),
                    })
                    raise ReActAgentError(
                        f"Malformed ReAct tool call: {error}",
                        trace=trace,
                        retrieved=list(retrieved_by_id.values()),
                    ) from error

                normalized = query.casefold().strip()
                if reject_duplicate_queries and normalized in seen_queries:
                    observation = f'Query "{query}" was already searched. Use another query.'
                    trace.append({
                        "action": "search",
                        "iteration": iteration,
                        "query": query,
                        "embedded_query": query,
                        "status": "duplicate_rejected",
                        "num_results": 0,
                        "result_ids": [],
                        "delivery": {"delivered": [], "dropped": [], "failed": [], "drop_reasons": {}},
                    })
                else:
                    seen_queries.add(normalized)
                    try:
                        search_output = search_fn(query)
                    except Exception as error:
                        trace.append({
                            "action": "search",
                            "iteration": iteration,
                            "query": query,
                            "status": "error",
                            "exception_class": type(error).__name__,
                            "error": str(error),
                        })
                        raise ReActAgentError(
                            f"ReAct search failed: {error}",
                            trace=trace,
                            retrieved=list(retrieved_by_id.values()),
                        ) from error
                    if isinstance(search_output, dict):
                        search_results = list(search_output.get("results", []))
                        observation = str(search_output.get("observation", ""))
                        delivery = dict(search_output.get("delivery", {}))
                        embedded_query = str(search_output.get("embedded_query", query))
                    else:
                        search_results = []
                        observation = str(search_output)
                        delivery = {}
                        embedded_query = query
                    result_ids: list[str] = []
                    for result in search_results:
                        chunk_id = str(result.get("chunk_id") or result.get("id") or "")
                        key = chunk_id or f"anonymous:{len(retrieved_by_id)}"
                        if chunk_id:
                            result_ids.append(chunk_id)
                        retrieved_by_id.setdefault(key, dict(result))
                    trace.append({
                        "action": "search",
                        "iteration": iteration,
                        "query": query,
                        "embedded_query": embedded_query,
                        "status": "ok",
                        "num_results": len(search_results),
                        "result_ids": result_ids,
                        "results": search_results,
                        "delivery": delivery,
                        "parsed_action": "search_memory",
                    })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": observation or "(no results found)",
                })
            continue

        call_trace["parsed_action"] = "answer"
        answer = str(getattr(message, "content", "") or "").strip()
        if not answer:
            trace.append({
                "action": "malformed_answer",
                "iteration": iteration,
                "status": "error",
                "error": "empty terminal answer",
            })
            raise ReActAgentError(
                f"ReAct iteration {iteration} returned an empty terminal answer",
                trace=trace,
                retrieved=list(retrieved_by_id.values()),
            )
        trace.append({
            "action": "answer",
            "iteration": iteration,
            "status": "ok",
            "predicted": answer,
            "forced": False,
            "parsed_action": "answer",
        })
        return answer, trace, list(retrieved_by_id.values())

    messages.append({
        "role": "user",
        "content": "Search budget exhausted. Return the best short answer from retrieved evidence, or exactly I don't know.",
    })
    forced_kwargs = {
        "model": reasoner_config.get("llm_model", "gpt-4o"),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 200,
    }
    _apply_extra_body(forced_kwargs, reasoner_config)
    message, call_trace, last_error = _react_chat_completion(
        client,
        forced_kwargs,
        iteration=max_iterations + 1,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        require_content=True,
    )
    call_trace["forced"] = True
    call_trace["parsed_action"] = "answer" if message is not None else None
    trace.append(call_trace)
    if message is None:
        raise ReActAgentError(
            f"Forced ReAct answer exhausted retries: {last_error}",
            trace=trace,
            retrieved=list(retrieved_by_id.values()),
        )
    answer = str(getattr(message, "content", "") or "").strip()
    trace.append({
        "action": "answer",
        "iteration": max_iterations + 1,
        "status": "ok",
        "predicted": answer,
        "forced": True,
        "parsed_action": "answer",
    })
    return answer, trace, list(retrieved_by_id.values())


def react_agent_multimodal(
    question: str,
    search_fn,
    reasoner_config: dict,
    max_iterations: int = 5,
    max_images_per_obs: int = 3,
    max_audio_per_obs: int = 2,
    max_video_per_obs: int = 1,
    max_media_size_mb: float = 20,
    reject_duplicate_queries: bool = True,
) -> tuple[str, list[dict], list[dict]]:
    """Run marker-based ReAct with observable multimodal delivery."""
    client = create_openai_client(reasoner_config)

    system_prompt = (
        "You are a multimodal memory retrieval agent. A user asks a question "
        "about their conversation history. You can search memory for relevant "
        "information — search results may include text, images, audio, and "
        "video files.\n\n"
        "To search, write exactly:\n"
        "SEARCH: <your query>\n\n"
        "To answer, write exactly:\n"
        "ANSWER: <your answer>\n\n"
        "Rules:\n"
        "- If the question can be answered immediately, write ANSWER right "
        "away. Otherwise, search first.\n"
        "- Write only ONE SEARCH or ONE ANSWER per response.\n"
        "- Search with short keyword queries (2-5 words), not full sentences.\n"
        "- Look carefully at images and listen to audio — they may contain "
        "visual or auditory details the text does not describe.\n"
        "- After 2-3 searches, synthesize an answer from what you've found.\n"
        "- Answer using ONLY the retrieved evidence. Do not use outside "
        "knowledge.\n"
        "- Your answer should be as short as possible — typically a single "
        "phrase, number, name, or date.\n"
        "- Do not include reasoning, explanations, or hedging in your final "
        "ANSWER. No markdown, no markdown fences.\n"
        f"- {evidence_update_answer_rule()}\n"
        "- For knowledge_update questions (what changed / was updated), answer "
        "with the full transition as \"from <old> to <new>\", not only the "
        "final state.\n"
        "- Do NOT repeat the same search query — each search should be "
        "different from previous ones.\n"
        "- Keep searching with different queries until you have enough "
        "evidence. Try different angles if earlier searches miss.\n"
        "- The question may require multiple pieces of evidence (multi-hop). "
        "If one search doesn't find a specific detail, search for a "
        "DIFFERENT piece of evidence instead — you can come back to the "
        "missing detail later. Don't get stuck repeating the same angle.\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    trace: list[dict] = []
    all_retrieved: list[dict] = []
    seen_queries: set[str] = set()
    max_retries = max(1, int(reasoner_config.get("max_retries", 2)))
    retry_backoff = float(reasoner_config.get("retry_backoff", 2.0))
    marker_nudges = 0
    max_marker_nudges = int(reasoner_config.get("max_marker_nudges", 2))

    for iteration in range(1, max_iterations + 1):
        kwargs: dict = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": messages,
            "temperature": reasoner_config.get("temperature", 0.0),
            "max_tokens": reasoner_config.get("max_tokens", 1024),
        }
        _apply_extra_body(kwargs, reasoner_config)
        message, call_trace, last_error = _react_chat_completion(
            client,
            kwargs,
            iteration=iteration,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            require_content=True,
        )
        trace.append(call_trace)
        if message is None:
            raise ReActAgentError(
                f"ReAct-MM iteration {iteration} exhausted retries: {last_error}",
                trace=trace,
                retrieved=all_retrieved,
            )
        text = str(getattr(message, "content", "") or "").strip()
        logger.debug(f"[ReAct-MM Iter {iteration}] raw response:\n{text}")

        answer_match = _parse_answer(text)
        query = _parse_search(text)
        if answer_match is not None and query is not None:
            call_trace["parsed_action"] = "malformed_multiple_actions"
            trace.append({
                "action": "malformed_marker_response",
                "iteration": iteration,
                "status": "error",
                "response": text,
                "error": "response contained both SEARCH and ANSWER markers",
            })
            raise ReActAgentError(
                "ReAct-MM response contained both SEARCH and ANSWER markers",
                trace=trace,
                retrieved=all_retrieved,
            )
        if answer_match is not None:
            call_trace["parsed_action"] = "answer"
            reasoning = _parse_reasoning(text)
            trace.append({
                "action": "answer",
                "iteration": iteration,
                "status": "ok",
                "predicted": answer_match,
                "reasoning": reasoning,
                "forced": False,
                "parsed_action": "answer",
            })
            return answer_match, trace, all_retrieved

        if query is not None:
            call_trace["parsed_action"] = "search_memory"
            reasoning = _parse_reasoning(text)
            if reject_duplicate_queries:
                q_norm = query.casefold().strip()
                if q_norm in seen_queries:
                    trace.append({
                        "action": "search",
                        "iteration": iteration,
                        "query": query,
                        "embedded_query": query,
                        "status": "duplicate_rejected",
                        "num_results": 0,
                        "top_score": 0.0,
                        "result_ids": [],
                        "delivery": {"delivered": [], "dropped": [], "failed": [], "drop_reasons": {}},
                    })
                    messages.append({"role": "assistant", "content": text})
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text",
                             "text": f'You already searched for "{query}". '
                                     f"Try a different query or answer based "
                                     f"on what you have."},
                        ],
                    })
                    continue
                seen_queries.add(q_norm)

            try:
                search_output = search_fn(query)
            except Exception as error:
                trace.append({
                    "action": "search",
                    "iteration": iteration,
                    "query": query,
                    "status": "error",
                    "exception_class": type(error).__name__,
                    "error": str(error),
                })
                raise ReActAgentError(
                    f"ReAct-MM search failed: {error}",
                    trace=trace,
                    retrieved=all_retrieved,
                ) from error
            if isinstance(search_output, dict):
                results = list(search_output.get("results", []))
                embedded_query = str(search_output.get("embedded_query", query))
                observation = search_output.get("observation")
                delivery = dict(search_output.get("delivery", {}))
            else:
                results = list(search_output)
                embedded_query = query
                observation = None
                delivery = {}
            existing_ids = {
                str(unit.get("chunk_id") or unit.get("id") or "")
                for unit in all_retrieved
            }
            for result in results:
                chunk_id = str(result.get("chunk_id") or result.get("id") or "")
                if chunk_id and chunk_id not in existing_ids:
                    all_retrieved.append(result)
                    existing_ids.add(chunk_id)

            n_results = len(results)
            top_score = round(results[0]["score"], 4) if results else 0.0
            if observation is None:
                observation, delivery = _build_multimodal_observation(
                    results, max_images_per_obs, max_audio_per_obs,
                    max_video_per_obs, max_media_size_mb,
                    api_format=reasoner_config.get("api_format", "openai"),
                )
            if not isinstance(observation, list):
                raise ReActAgentError(
                    "ReAct-MM search observation must be a content-block list",
                    trace=trace,
                    retrieved=all_retrieved,
                )
            attachment_counts = delivery.get("attachment_counts", {})
            trace.append({
                "action": "search",
                "iteration": iteration,
                "query": query,
                "embedded_query": embedded_query,
                "status": "ok",
                "num_results": n_results,
                "top_score": top_score,
                "result_ids": [
                    str(result.get("chunk_id") or result.get("id") or "")
                    for result in results
                ],
                "results": results,
                "delivery": delivery,
                "attached_images": attachment_counts.get("image", 0),
                "attached_audio": attachment_counts.get("audio", 0),
                "attached_video": attachment_counts.get("video", 0),
                "reasoning": reasoning,
                "parsed_action": "search_memory",
            })

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": observation})

            for _i in range(1, len(messages) - 2):
                _msg = messages[_i]
                if _msg["role"] == "user" and isinstance(_msg["content"], list):
                    _msg["content"] = [
                        b for b in _msg["content"]
                        if b.get("type") == "text"
                    ] or [{"type": "text", "text": "[media already shown]"}]
            continue

        call_trace["parsed_action"] = "malformed"
        if marker_nudges < max_marker_nudges:
            # Corrective nudge instead of a hard failure: only episodes that
            # would previously have errored take this path, so completed
            # items are unaffected. The nudge is recorded in the trace.
            marker_nudges += 1
            trace.append({
                "action": "malformed_marker_response",
                "iteration": iteration,
                "status": "nudged",
                "nudge_number": marker_nudges,
                "response": text,
            })
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": (
                    "Your reply did not contain a valid action. Reply with "
                    "exactly one line starting with SEARCH: <query> to search "
                    "again, or ANSWER: <answer> to give your final answer."
                ),
            })
            continue
        trace.append({
            "action": "malformed_marker_response",
            "iteration": iteration,
            "status": "error",
            "response": text,
        })
        raise ReActAgentError(
            "ReAct-MM response contained neither SEARCH nor ANSWER marker",
            trace=trace,
            retrieved=all_retrieved,
        )

    messages.append({
        "role": "user",
        "content": (
            "You have reached the maximum number of searches. "
            'Write ANSWER: followed by your best answer, or ANSWER: I don\'t know.'
        ),
    })
    kwargs = {
        "model": reasoner_config.get("llm_model", "gpt-4o"),
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": int(reasoner_config.get("forced_answer_max_tokens", 512)),
    }
    _apply_extra_body(kwargs, reasoner_config)
    message, call_trace, last_error = _react_chat_completion(
        client,
        kwargs,
        iteration=max_iterations + 1,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        require_content=True,
    )
    call_trace["forced"] = True
    trace.append(call_trace)
    if message is None:
        raise ReActAgentError(
            f"Forced ReAct-MM answer exhausted retries: {last_error}",
            trace=trace,
            retrieved=all_retrieved,
        )
    text = str(getattr(message, "content", "") or "").strip()
    answer = _parse_answer(text)
    answer_without_marker = False
    if answer is None and text:
        # The forced prompt demands an ANSWER: marker, but some models emit
        # the answer directly. Accept the trimmed text rather than discarding
        # a real prediction; the trace records the leniency.
        answer = text
        answer_without_marker = True
    if answer is None:
        call_trace["parsed_action"] = "malformed"
        trace.append({
            "action": "malformed_marker_response",
            "iteration": max_iterations + 1,
            "status": "error",
            "response": text,
            "forced": True,
        })
        raise ReActAgentError(
            "Forced ReAct-MM response did not contain an ANSWER marker",
            trace=trace,
            retrieved=all_retrieved,
        )
    call_trace["parsed_action"] = "answer"
    trace.append({
        "action": "answer",
        "iteration": max_iterations + 1,
        "status": "ok",
        "predicted": answer,
        "forced": True,
        "answer_without_marker": answer_without_marker,
        "parsed_action": "answer",
    })
    return answer, trace, all_retrieved


# -- ReAct multimodal helpers -------------------------------------------------


def _parse_reasoning(text: str) -> str | None:
    """Extract reasoning text before any SEARCH:/ANSWER: marker."""
    import re
    # Take everything before the first SEARCH: or ANSWER: marker
    m = re.search(r'^(.*?)(?:SEARCH:|ANSWER:)', text, re.IGNORECASE | re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        return reasoning if reasoning else None
    return None


def _parse_search(text: str) -> str | None:
    """Extract query from ``SEARCH: <query>`` marker."""
    import re
    m = re.search(r'SEARCH:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _parse_answer(text: str) -> str | None:
    """Extract answer from ``ANSWER: <answer>`` marker."""
    import re
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def _load_media_b64(path: str, max_size_mb: float = 20) -> str | None:
    """Load an image/audio/video file and return as a base64 data URI."""
    import base64 as _b64
    from io import BytesIO
    import os as _os

    if not _os.path.exists(path):
        return None
    if _os.path.getsize(path) > max_size_mb * 1024 * 1024:
        return None
    try:
        ext = _os.path.splitext(path)[1].lower()
        # Images
        if ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
            from PIL import Image
            img = Image.open(path).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = _b64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        # Audio
        if ext in (".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".opus"):
            with open(path, "rb") as f:
                data = f.read()
            b64 = _b64.b64encode(data).decode("utf-8")
            mime = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                    ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                    ".flac": "audio/flac", ".aac": "audio/aac",
                    ".opus": "audio/opus"}.get(ext, "audio/mpeg")
            return f"data:{mime};base64,{b64}"
        # Video
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"):
            with open(path, "rb") as f:
                data = f.read()
            b64 = _b64.b64encode(data).decode("utf-8")
            mime = {".mp4": "video/mp4", ".avi": "video/x-msvideo",
                    ".mov": "video/quicktime", ".mkv": "video/x-matroska",
                    ".webm": "video/webm", ".flv": "video/x-flv",
                    ".wmv": "video/x-ms-wmv"}.get(ext, "video/mp4")
            return f"data:{mime};base64,{b64}"
        return None
    except Exception:
        return None


def _build_multimodal_observation(
    results: list[dict],
    max_images: int = 3,
    max_audio: int = 2,
    max_video: int = 1,
    max_media_size_mb: float = 20,
    api_format: str = "openai",
) -> tuple[list[dict], dict]:
    """Build an observation and its explicit common delivery manifest."""
    packed = pack_evidence(
        results,
        DeliveryPolicy(
            mode="production_budget",
            max_images=max_images,
            max_audio=max_audio,
            max_video=max_video,
            max_media_size_mb=max_media_size_mb,
            attach_media=True,
        ),
    )
    packed = materialize_openai_media_blocks(packed, api_format=api_format)

    text_parts = ["Search results (most relevant first):\n"]
    for i, r in enumerate(packed["delivered"]):
        text_parts.append(
            f"[{i + 1}] score={r.get('score', 0):.3f} | "
            f"{r.get('timestamp', '')} | {r.get('modality', 'text')}\n"
            f"    {r.get('content', '')}"
        )
    if packed["provider_blocks"]:
        text_parts.append("\nAttached media files are shown below.")
    text_blob = "\n\n".join(text_parts)
    content: list[dict] = [
        {"type": "text", "text": text_blob},
        *packed["provider_blocks"],
    ]
    return content, packed


def print_console_summary(summary: dict):
    """Print a compact summary table to the console."""
    print("\n" + "=" * 72)
    print("EVALUATION SUMMARY")
    print("=" * 72)
    gm = summary.get("global_metrics", summary)
    print(f"Total questions:   {gm.get('total_questions', 0)}")
    print(f"Overall accuracy:  {gm.get('overall_accuracy', 0):.3f}")
    print(f"  Answerable:      {gm.get('accuracy_answerable', 0):.3f} "
          f"({gm.get('correct_answerable', 0)} / {gm.get('answerable_questions', 0)})")
    print(f"  Unanswerable:    {gm.get('accuracy_unanswerable', 0):.3f} "
          f"({gm.get('correct_unanswerable', 0)} / {gm.get('unanswerable_questions', 0)})")

    per_type = gm.get("per_question_type", {})
    if per_type:
        print(f"\n{'Question Type':<30s} {'Accuracy':>8s}  {'Count':>5s}")
        print("-" * 48)
        for qtype, stats in per_type.items():
            print(f"  {qtype:<28s} {stats['accuracy']:>8.3f}  {stats['count']:>5d}")

    print("=" * 72)
