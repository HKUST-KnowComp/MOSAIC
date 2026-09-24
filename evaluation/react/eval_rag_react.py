#!/usr/bin/env python3
"""**ReAct-RAG** — Evaluate iterative retrieval on any of the 3 FAISS indexes.

The LLM decides how many searches to run and when it has enough information,
using a ``search_memory`` tool to query the FAISS index. Works with:

- Caption→Text / Omni→Text indexes  (text embedding + text LLM reasoner)
- Omni-MM index                      (jina embedding + omni VL reasoner)

Usage::

    python evaluation/eval_rag_react.py
    python evaluation/eval_rag_react.py --max-personas 2
    python evaluation/eval_rag_react.py --config evaluation/config_eval_react.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path wrangling
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from _common import (
    load_config,
    get_text_embedding_model,
    get_omni_embed_model,
    encode_text_batch,
    encode_omni_queries,
    load_faiss_index,
    search_faiss_index_filtered,
    ReActAgentError,
    react_agent,
    react_agent_multimodal,
    rewrite_query,
    match_answer,
    compute_retrieval_recall,
    aggregate_results,
    save_results,
    print_console_summary,
    _apply_extra_body,
    call_llm_with_retry,
    create_openai_client,
    load_failed_entries,
)
from protocol import (
    DeliveryPolicy,
    QueryManifestCache,
    load_canonical_records,
    make_item_trace,
    materialize_openai_media_blocks,
    overlay_persona_items,
    pack_evidence,
    records_by_persona,
    retrieval_metric_aggregate,
    validate_result_qids,
    write_run_manifest,
)
from rag.eval_rag_mm import call_omni_reasoner
from rag.eval_rag_text import call_text_reasoner

PROJECT_ROOT = os.path.dirname(HERE)

FALLBACK_DATE_FMT = "%Y/%m/%d"


# ---------------------------------------------------------------------------
# Time-cutoff filter
# ---------------------------------------------------------------------------


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.split(" (")[0], FALLBACK_DATE_FMT)
    except (ValueError, TypeError):
        return None


def filter_by_time(chunks: list[dict], question_date: str) -> list[dict]:
    q_date = _parse_date(question_date)
    if q_date is None:
        return chunks
    return [c for c in chunks
            if _parse_date(c.get("timestamp", "")) is None
            or _parse_date(c.get("timestamp", "")) <= q_date]


# ---------------------------------------------------------------------------
# Index type detection
# ---------------------------------------------------------------------------


def _is_mm_index(manifest: dict) -> bool:
    """Return True if the index was built with a multimodal embedding model."""
    model = (manifest.get("model_name") or "").lower()
    return "jina" in model or "omni" in model or "clip" in model


# ---------------------------------------------------------------------------
# Text search function (Caption→Text, Omni→Text)
# ---------------------------------------------------------------------------


def _make_text_search_fn(
    index, metadata,
    embedding_model_name: str,
    embedding_device: str,
    top_k: int,
    question_date: str,
    enable_time_filter: bool,
    rewrite_config: dict | None = None,
    observation_max_results: int | None = None,
    observation_max_chars: int | None = None,
    score_threshold: float = 0.0,
    candidate_k: int | None = None,
):
    """Return a search_memory function for text FAISS indexes.

    If *rewrite_config* is provided, each query is rewritten into keywords
    before embedding (same as RAG's query_rewrite).
    """

    def search(query: str) -> dict:
        original_query = query
        if rewrite_config:
            query = rewrite_query(query, rewrite_config)
        q_emb = encode_text_batch(
            [query], embedding_model_name, embedding_device,
            prompt_name="query",
        ).astype("float32")
        def _eligible(unit: dict) -> bool:
            if enable_time_filter and not filter_by_time([unit], question_date):
                return False
            return unit.get("score", 0) >= score_threshold

        results = search_faiss_index_filtered(
            index,
            q_emb,
            metadata,
            k=top_k,
            candidate_k=candidate_k or top_k,
            eligibility_fn=_eligible,
        )
        observation, delivery = _format_search_results_text(
            results,
            max_results=observation_max_results,
            max_content_chars=observation_max_chars,
        )
        return {
            "query": original_query,
            "embedded_query": query,
            "results": results,
            "observation": observation,
            "delivery": delivery,
        }

    return search


def _format_search_results_text(
    results: list[dict],
    *,
    max_results: int | None = None,
    max_content_chars: int | None = None,
) -> tuple[str, dict]:
    if not results:
        return "(no results found)", {
            "delivered": [], "dropped": [], "failed": [],
            "drop_reasons": {}, "attachment_counts": {},
            "effective_context_chars": 0,
        }
    visible = results if max_results is None else results[:max_results]
    parts = []
    delivered = []
    for i, r in enumerate(visible):
        content = str(r.get("content", ""))
        truncated = max_content_chars is not None and len(content) > max_content_chars
        rendered_content = content[:max_content_chars] if truncated else content
        parts.append(
            f"[{i + 1}] score={r.get('score', 0):.3f} | "
            f"{r.get('timestamp', '')} | {r.get('modality', 'text')}\n"
            f"    {rendered_content}"
        )
        delivered_unit = dict(r)
        delivered_unit.update({
            "observation_truncated": truncated,
            "delivered_chars": len(rendered_content),
            "original_chars": len(content),
            "content": rendered_content,
            "native_attached": False,
            "delivery_status": "text_truncated" if truncated else "text_only",
        })
        if truncated:
            delivered_unit["fact_key"] = None
            delivered_unit["fact_keys"] = []
        delivered.append(delivered_unit)
    dropped = [
        {
            **dict(result),
            "drop_reason": "observation_truncation",
            "native_attached": False,
            "delivery_status": "dropped",
        }
        for result in results[len(visible):]
    ]
    truncated_losses = [
        {
            **dict(result),
            "drop_reason": "content_truncation",
            "native_attached": False,
            "delivery_status": "partial",
        }
        for result, delivered_unit in zip(visible, delivered)
        if delivered_unit["observation_truncated"]
    ]
    dropped.extend(truncated_losses)
    reasons = Counter(unit["drop_reason"] for unit in dropped)
    return "\n\n".join(parts), {
        "delivered": delivered,
        "dropped": dropped,
        "failed": [],
        "drop_reasons": dict(reasons),
        "attachment_counts": {},
        "effective_context_chars": sum(unit["delivered_chars"] for unit in delivered),
    }


# ---------------------------------------------------------------------------
# Multimodal search function (Omni-MM)
# ---------------------------------------------------------------------------


def _make_mm_search_fn(
    index, metadata,
    omni_config: dict,
    top_k: int,
    question_date: str,
    enable_time_filter: bool,
    rewrite_config: dict | None = None,
    score_threshold: float = 0.0,
    candidate_k: int | None = None,
):
    """Return a search_memory function for multimodal FAISS indexes.

    Returns raw result dicts (with media_path) so react_agent_multimodal()
    can attach actual media files to the observation.

    If *rewrite_config* is provided, each query is rewritten into keywords
    before embedding.
    """

    def search(query: str) -> dict:
        original_query = query
        if rewrite_config:
            query = rewrite_query(query, rewrite_config)
        q_emb = encode_omni_queries([query], omni_config).astype("float32")
        def _eligible(unit: dict) -> bool:
            if enable_time_filter and not filter_by_time([unit], question_date):
                return False
            return unit.get("score", 0) >= score_threshold

        results = search_faiss_index_filtered(
            index,
            q_emb,
            metadata,
            k=top_k,
            candidate_k=candidate_k or top_k,
            eligibility_fn=_eligible,
        )
        return {
            "query": original_query,
            "embedded_query": query,
            "results": results,
        }

    return search


# ---------------------------------------------------------------------------
# Omni-MM final answer with media
# ---------------------------------------------------------------------------


def _load_image_base64(path: str, max_size_mb: float = 20) -> str | None:
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) > max_size_mb * 1024 * 1024:
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _load_audio_base64(path: str, max_size_mb: float = 20) -> str | None:
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) > max_size_mb * 1024 * 1024:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime_map = {".mp3": "audio/mpeg", ".wav": "audio/wav",
                     ".ogg": "audio/ogg", ".m4a": "audio/mp4",
                     ".flac": "audio/flac", ".aac": "audio/aac"}
        return f"data:{mime_map.get(ext, 'audio/mpeg')};base64,{b64}"
    except Exception:
        return None


def _load_video_base64(path: str, max_size_mb: float = 20) -> str | None:
    """Load a video file and return as data:video/...;base64,... string."""
    if not os.path.exists(path):
        return None
    if os.path.getsize(path) > max_size_mb * 1024 * 1024:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".mp4": "video/mp4", ".avi": "video/x-msvideo",
                ".mov": "video/quicktime", ".mkv": "video/x-matroska",
                ".webm": "video/webm", ".flv": "video/x-flv",
                ".wmv": "video/x-ms-wmv"}.get(ext, "video/mp4")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _mm_final_answer(
    question: str,
    retrieved: list[dict],
    reasoner_config: dict,
    max_media_size_mb: float = 20,
    max_images: int = 5,
    max_audio: int = 3,
    max_video: int = 1,
    question_type: str | None = None,
) -> str:
    """After ReAct retrieval, make a final multimodal call with attached media."""
    text_parts = []
    for i, c in enumerate(retrieved):
        text_parts.append(
            f"[{i + 1}] score={c.get('score', 0):.3f} | "
            f"{c.get('timestamp', '')} | {c.get('modality', 'text')}\n"
            f"    {c.get('content', '')}"
        )
    context_str = "\n\n".join(text_parts)

    from _common import evidence_update_answer_rule

    system_prompt = (
        "You are an evaluation assistant. Answer questions about a user "
        "based on retrieved conversation history and attached media.\n\n"
        "Answer rules:\n"
        "1. Answer using ONLY the retrieved evidence provided. Do not use "
        "outside knowledge.\n"
        "2. The evidence may include native multimodal artefacts (images, "
        "audio recordings, video clips). Examine each artefact carefully — "
        "they may contain visual or auditory details that the surrounding "
        "text does not describe.\n"
        "3. Your answer should be as short as possible — typically a single "
        "phrase, number, name, or date.\n"
        "4. Do not include reasoning, explanations, or hedging in your final "
        "answer. Do not describe the multimodal artefacts in your answer; "
        "extract the requested fact directly.\n"
        "5. If the retrieved evidence does not contain enough information to "
        'answer the question, respond with exactly: I don\'t know.\n'
        f"6. {evidence_update_answer_rule(question_type)}\n"
        '7. For temporal references in the question ("currently", "now", '
        '"recently"), interpret them relative to the question\'s date.\n'
        "8. Examples of correct answer format:\n"
        '   - Question: "What was the work start time?" → Answer: "8:15 AM"\n'
        '   - Question: "What did the charity race raise awareness for?" → Answer: "mental health"\n'
        '   - Question: "How many books did I read this year?" → Answer: "12"\n'
    )

    user_text = (
        "<RETRIEVED CONVERSATION HISTORY>\n"
        f"{context_str}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    content_parts = [{"type": "text", "text": user_text}]

    n_img = 0
    n_aud = 0
    n_vid = 0
    for c in retrieved:
        mp = c.get("media_path")
        mod = c.get("modality", "text")
        if not mp:
            continue
        if mod == "image" and n_img < max_images:
            b64 = _load_image_base64(mp, max_media_size_mb)
            if b64:
                content_parts.append({"type": "image_url", "image_url": {"url": b64}})
                n_img += 1
        elif mod == "audio" and n_aud < max_audio:
            b64 = _load_audio_base64(mp, max_media_size_mb)
            if b64:
                content_parts.append({
                    "type": "input_audio",
                    "input_audio": {"data": b64.split(",", 1)[-1], "format": "mp3"},
                })
                n_aud += 1
        elif mod == "video" and n_vid < max_video:
            b64 = _load_video_base64(mp, max_media_size_mb)
            if b64:
                if reasoner_config.get("api_format") == "gemini":
                    content_parts.append({"type": "image_url", "image_url": {"url": b64}})
                else:
                    content_parts.append({"type": "video_url", "video_url": {"url": b64}})
                n_vid += 1

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content_parts},
    ]

    try:
        client = create_openai_client(reasoner_config)

        # Log text parts (skip base64 media)
        text_log = "\n\n".join(
            p["text"] for p in content_parts if p["type"] == "text"
        )
        media_log = f"attached_media: {n_img} images, {n_aud} audio, {n_vid} video"
        logger.debug(
            f"[MM Final Answer Prompt]\nsystem: {system_prompt}\n\n"
            f"user: {text_log}\n{media_log}"
        )

        kwargs = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": messages,
            "temperature": reasoner_config.get("temperature", 0.0),
            "max_tokens": 200,
        }
        _apply_extra_body(kwargs, reasoner_config)
        answer = call_llm_with_retry(client, kwargs)
        logger.debug(f"[MM Final Answer Output] {answer}")
        return answer
    except Exception as e:
        logger.warning(f"MM final answer call failed: {e}")
        return "ERROR: " + str(e)


# ---------------------------------------------------------------------------
# Per-persona evaluation
# ---------------------------------------------------------------------------


def _unit_key(unit: dict, fallback: str) -> str:
    return str(unit.get("chunk_id") or unit.get("id") or fallback)


def _merge_react_delivery(trace: list[dict]) -> dict:
    """Merge per-search observation manifests into first-seen delivered context."""
    delivered_by_id: dict[str, dict] = {}
    dropped: list[dict] = []
    failed: list[dict] = []
    reasons: Counter[str] = Counter()
    mode = "production_budget"
    for event_index, event in enumerate(trace):
        if event.get("action") != "search" or event.get("status") != "ok":
            continue
        delivery = event.get("delivery") or {}
        mode = str(delivery.get("mode", mode))
        for unit_index, raw_unit in enumerate(delivery.get("delivered", [])):
            unit = dict(raw_unit)
            key = _unit_key(unit, f"delivered:{event_index}:{unit_index}")
            previous = delivered_by_id.get(key)
            previous_facts = (previous or {}).get("fact_keys") or (previous or {}).get("fact_key")
            current_facts = unit.get("fact_keys") or unit.get("fact_key")
            if previous is None or (not previous_facts and current_facts):
                delivered_by_id[key] = unit
        for collection_name, target in (("dropped", dropped), ("failed", failed)):
            for raw_unit in delivery.get(collection_name, []):
                unit = dict(raw_unit)
                unit["react_iteration"] = event.get("iteration")
                unit["react_query"] = event.get("query")
                target.append(unit)
                reason = str(unit.get("drop_reason") or collection_name)
                reasons[reason] += 1
    delivered = list(delivered_by_id.values())
    return {
        "mode": mode,
        "provider_blocks": [],
        "delivered": delivered,
        "dropped": dropped,
        "failed": failed,
        "drop_reasons": dict(reasons),
        "effective_context_chars": sum(
            len(str(unit.get("content") or "")) for unit in delivered
        ),
        "attachment_counts": {
            modality: sum(
                1 for unit in delivered
                if unit.get("modality") == modality and unit.get("native_attached")
            )
            for modality in ("audio", "image", "video")
        },
    }


def _empty_evaluation() -> dict:
    return {
        "exact_match": False,
        "alias_match": False,
        "unanswerable_correct": False,
    }


def _terminal_error_record(
    item: dict,
    persona_uuid: str,
    error: Exception | str,
    query_info: dict | None = None,
    *,
    react_trace: list[dict] | None = None,
    retrieved: list[dict] | None = None,
    react_mode: str = "adaptive_react",
) -> dict:
    message = str(error)
    error_class = type(error).__name__ if isinstance(error, Exception) else "Error"
    api_status = "timeout" if "timeout" in f"{error_class} {message}".lower() else "error"
    question_id = item.get("question_id", "unknown")
    query_info = query_info or {
        "original": item.get("question", ""),
        "retrieval_query": item.get("question", ""),
        "rewrite_model": None,
    }
    react_trace = list(react_trace or [])
    retrieved = list(retrieved or [])
    delivery = _merge_react_delivery(react_trace)
    react_block = {
        "mode": react_mode,
        "events": react_trace,
        "num_searches": sum(
            event.get("action") == "search" and event.get("status") == "ok"
            for event in react_trace
        ),
        "num_unique_retrieved": len(retrieved),
        "forced": False,
        "terminal_status": api_status,
    }
    trace = make_item_trace(
        question_id=question_id,
        query=query_info,
        ranked=retrieved,
        evidence_plan=item.get("evidence_plan", []),
        delivery=delivery,
        answer={
            "predicted": "",
            "api_status": api_status,
            "forced": False,
            "error_class": error_class,
            "error_message": message,
        },
        react=react_block,
        required_session_ids=item.get("answer_session_ids", []),
    )
    return {
        "question_id": question_id,
        "question_type": item.get("question_type", "unknown"),
        "answerability": item.get("answerability", "unknown"),
        "question": item.get("question", ""),
        "expected_answer": item.get("answer", ""),
        "predicted_answer": "",
        "evaluation": _empty_evaluation(),
        "error": message,
        "api_status": api_status,
        "retrieval_recall": compute_retrieval_recall(
            retrieved, item.get("evidence_plan", [])
        ),
        "react_trace": react_trace,
        "trace": trace,
        "persona_uuid": persona_uuid,
    }


def _parity_answer(
    question: str,
    retrieved: list[dict],
    *,
    is_mm: bool,
    reasoner_config: dict,
    react_config: dict,
    question_type: str | None = None,
) -> tuple[str, dict]:
    """Use exactly the Naive RAG packer and reader for the parity gate."""
    if is_mm:
        delivery_config = dict(react_config)
        delivery_config.setdefault(
            "max_images_per_query", react_config.get("max_images_per_obs", 3)
        )
        delivery_config.setdefault(
            "max_audio_per_query", react_config.get("max_audio_per_obs", 2)
        )
        delivery_config.setdefault(
            "max_video_per_query", react_config.get("max_video_per_obs", 1)
        )
        packed = pack_evidence(
            retrieved, DeliveryPolicy.from_config(delivery_config)
        )
        packed = materialize_openai_media_blocks(
            packed, api_format=reasoner_config.get("api_format")
        )
        predicted = call_omni_reasoner(
            question,
            packed["delivered"],
            reasoner_config,
            packed["provider_blocks"],
            question_type=question_type,
        )
        return predicted, packed

    packed = pack_evidence(
        retrieved,
        DeliveryPolicy(
            mode="production_budget",
            attach_media=False,
            max_context_chars=react_config.get("max_context_chars"),
        ),
    )
    return call_text_reasoner(
        question, packed["delivered"], reasoner_config, question_type=question_type,
    ), packed


def evaluate_persona_react(
    persona_entry: dict,
    index_dir: str,
    is_mm: bool,
    embedding_config: dict,
    react_reasoner_config: dict,
    answer_reasoner_config: dict,
    react_config: dict,
    canonical_records: list[dict],
    query_cache: QueryManifestCache,
    only_qids: set[str] | None = None,
) -> list[dict]:
    """Load a persona's index and evaluate questions with ReAct.

    If *only_qids* is provided, only those question_ids are evaluated.
    """
    persona_uuid = persona_entry["persona_uuid"]
    persona_path = persona_entry["path"]
    react_mode = react_config.get("react_mode", "adaptive_react")
    if react_mode not in {"adaptive_react", "one_search_parity"}:
        raise ValueError(f"Unknown react_mode: {react_mode}")

    index_items: list[dict] = []
    try:
        index, metadata = load_faiss_index(
            os.path.join(persona_path, "index.faiss"),
            os.path.join(persona_path, "metadata.json"),
        )
        with open(os.path.join(persona_path, "items.json")) as f:
            index_items = json.load(f)
    except Exception as error:
        logger.exception(f"Failed to load index for {persona_uuid}")
        items = overlay_persona_items([], canonical_records, only_qids=only_qids)
        return [
            _terminal_error_record(
                item, persona_uuid, error, react_mode=react_mode
            )
            for item in items
        ]

    items = overlay_persona_items(
        index_items, canonical_records, only_qids=only_qids
    )

    max_iterations = react_config.get("max_iterations", 5)
    search_top_k = react_config.get("search_top_k", 10)
    enable_time_filter = react_config.get("enable_time_filter", False)
    score_threshold = float(react_config.get("score_threshold", 0.0))
    candidate_k = int(
        react_config.get(
            "candidate_k",
            search_top_k
            if not enable_time_filter and score_threshold <= 0
            else max(search_top_k * 2, search_top_k),
        )
    )
    enable_query_rewrite = bool(
        react_config.get("tool_query_rewrite", react_config.get("query_rewrite", False))
    ) and react_mode == "adaptive_react"
    reject_duplicate_queries = react_config.get("reject_duplicate_queries", True)
    rewrite_reasoner = react_config.get("rewrite_reasoner")
    rewrite_config = (rewrite_reasoner or react_reasoner_config) if enable_query_rewrite else None

    emb_model = embedding_config.get("model_name", "?")
    emb_device = embedding_config.get("device", "cpu")

    results = []
    for item in tqdm(items, desc=f"  {persona_uuid[:8]}", unit="q", leave=False):
        question_id = item.get("question_id", "unknown")
        if only_qids is not None and question_id not in only_qids:
            continue
        question = item.get("question", "")
        question_type = item.get("question_type", "unknown")
        answerability = item.get("answerability", "answerable")
        expected = item.get("answer", "")
        aliases = item.get("answer_aliases", [])
        question_date = item.get("question_date", "")

        t0 = time.time()
        query_info = {
            "original": question,
            "retrieval_query": question,
            "rewrite_model": None,
            "query_policy": "adaptive_react",
        }
        react_trace: list[dict] = []
        all_retrieved: list[dict] = []

        try:
            if is_mm:
                search_fn = _make_mm_search_fn(
                    index, metadata, embedding_config,
                    search_top_k, question_date, enable_time_filter,
                    rewrite_config=rewrite_config,
                    score_threshold=score_threshold,
                    candidate_k=candidate_k,
                )
            else:
                search_fn = _make_text_search_fn(
                    index, metadata, emb_model, emb_device,
                    search_top_k, question_date, enable_time_filter,
                    rewrite_config=rewrite_config,
                    observation_max_results=react_config.get("observation_max_results"),
                    observation_max_chars=react_config.get("observation_max_chars"),
                    score_threshold=score_threshold,
                    candidate_k=candidate_k,
                )

            if react_mode == "one_search_parity":
                rewrite_cfg = rewrite_reasoner or react_reasoner_config
                query_info = query_cache.resolve(
                    question_id,
                    question,
                    lambda value: rewrite_query(value, rewrite_cfg),
                )
                search_output = search_fn(query_info["retrieval_query"])
                all_retrieved = list(search_output.get("results", []))
                predicted, delivery = _parity_answer(
                    question,
                    all_retrieved,
                    is_mm=is_mm,
                    reasoner_config=answer_reasoner_config,
                    react_config=react_config,
                    question_type=question_type,
                )
                react_trace = [{
                    "action": "search",
                    "iteration": 1,
                    "query": query_info["retrieval_query"],
                    "embedded_query": search_output.get(
                        "embedded_query", query_info["retrieval_query"]
                    ),
                    "status": "ok",
                    "num_results": len(all_retrieved),
                    "result_ids": [
                        str(unit.get("chunk_id") or unit.get("id") or "")
                        for unit in all_retrieved
                    ],
                    "results": all_retrieved,
                    "delivery": delivery,
                    "parsed_action": "one_search_parity",
                }, {
                    "action": "answer",
                    "iteration": 1,
                    "status": (
                        "error" if predicted.startswith("ERROR:")
                        else "empty" if not predicted.strip()
                        else "ok"
                    ),
                    "predicted": predicted,
                    "forced": False,
                    "parsed_action": "answer",
                }]
                forced = False
            else:
                query_info = query_cache.resolve(question_id, question)
                if is_mm:
                    predicted, react_trace, all_retrieved = react_agent_multimodal(
                        question, search_fn, react_reasoner_config,
                        max_iterations=max_iterations,
                        max_images_per_obs=react_config.get("max_images_per_obs", 3),
                        max_audio_per_obs=react_config.get("max_audio_per_obs", 2),
                        max_video_per_obs=react_config.get("max_video_per_obs", 1),
                        max_media_size_mb=react_config.get("max_media_size_mb", 20),
                        reject_duplicate_queries=reject_duplicate_queries,
                    )
                else:
                    predicted, react_trace, all_retrieved = react_agent(
                        question, search_fn, react_reasoner_config,
                        max_iterations=max_iterations,
                        reject_duplicate_queries=reject_duplicate_queries,
                    )
                delivery = _merge_react_delivery(react_trace)
                answer_events = [
                    event for event in react_trace if event.get("action") == "answer"
                ]
                forced = bool(answer_events and answer_events[-1].get("forced"))

            if predicted.startswith("ERROR:"):
                api_status = "timeout" if "timeout" in predicted.lower() else "error"
            elif not predicted.strip():
                api_status = "empty"
            else:
                api_status = "ok"
            eval_result = (
                match_answer(predicted, expected, aliases, answerability)
                if api_status == "ok" else _empty_evaluation()
            )
            recall_metrics = compute_retrieval_recall(
                all_retrieved, item.get("evidence_plan", []),
            )

            evidence_plan_summary = [
                {
                    "fact_key": ev.get("fact_key"),
                    "fact_statement": ev.get("fact_statement"),
                    "modality": ev.get("modality"),
                    "modality_rationale": ev.get("modality_rationale"),
                    "required_for_answer": ev.get("required_for_answer"),
                    "role": ev.get("role"),
                }
                for ev in item.get("evidence_plan", [])
            ]

            react_block = {
                "mode": react_mode,
                "control_protocol": (
                    "one_search_parity" if react_mode == "one_search_parity"
                    else "marker" if is_mm else "function_calling"
                ),
                "events": react_trace,
                "num_searches": sum(
                    event.get("action") == "search" and event.get("status") == "ok"
                    for event in react_trace
                ),
                "num_unique_retrieved": len(all_retrieved),
                "forced": forced,
                "terminal_status": api_status,
            }
            item_trace = make_item_trace(
                question_id=question_id,
                query=query_info,
                ranked=all_retrieved,
                evidence_plan=item.get("evidence_plan", []),
                delivery=delivery,
                answer={
                    "predicted": predicted,
                    "api_status": api_status,
                    "forced": forced,
                },
                react=react_block,
                required_session_ids=item.get("answer_session_ids", []),
            )

            latency = time.time() - t0
            results.append({
                "question_id": question_id,
                "question_type": question_type,
                "answerability": answerability,
                "question": question,
                "expected_answer": expected,
                "answer_aliases": aliases,
                "chain_description": item.get("chain_description", ""),
                "multihop_chain": item.get("multihop_chain", ""),
                "evidence_plan": evidence_plan_summary,
                "predicted_answer": predicted,
                "api_status": api_status,
                "evaluation": eval_result,
                "retrieval_recall": recall_metrics,
                "react_trace": react_trace,
                "trace": item_trace,
                "total_iterations": react_block["num_searches"],
                "num_retrieved": len(all_retrieved),
                "retrieved": [
                    {
                        "chunk_id": r.get("chunk_id", ""),
                        "score": r.get("score", 0),
                        "modality": r.get("modality", "text"),
                        "content": (r.get("content", "") or "")[:300],
                        "session_id": r.get("session_id", ""),
                        "timestamp": r.get("timestamp", ""),
                        "media_path": r.get("media_path"),
                        "fact_key": r.get("fact_key"),
                        "fact_keys": r.get("fact_keys", []),
                        "pre_filter_rank": r.get("pre_filter_rank"),
                        "eligible_rank": r.get("eligible_rank"),
                    }
                    for r in all_retrieved
                ],
                "latency_seconds": round(latency, 3),
                "persona_uuid": persona_uuid,
            })

        except Exception as error:
            if isinstance(error, ReActAgentError):
                react_trace = error.trace
                all_retrieved = error.retrieved
            logger.warning(f"Failed to evaluate {question_id}: {error}")
            results.append(
                _terminal_error_record(
                    item,
                    persona_uuid,
                    error,
                    query_info,
                    react_trace=react_trace,
                    retrieved=all_retrieved,
                    react_mode=react_mode,
                )
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ReAct-RAG: Evaluate iterative retrieval on FAISS indexes"
    )
    p.add_argument(
        "--config", default=os.path.join(HERE, "config_eval_react.json"),
        help="Path to eval config JSON",
    )
    p.add_argument(
        "--max-personas", type=int, default=None,
        help="Limit number of personas",
    )
    p.add_argument(
        "--persona-filter", default=None,
        help="Comma-separated persona UUID substrings",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override output directory",
    )
    p.add_argument(
        "--query-workers", type=int, default=None,
        help="Parallel query workers (default from config, or 1)",
    )
    p.add_argument(
        "--react-mode",
        choices=("adaptive_react", "one_search_parity"),
        default=None,
        help="Override react_mode from config",
    )
    p.add_argument(
        "--resume", action="store_true", default=False,
        help="Resume from existing details.jsonl — re-evaluate only failed entries",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.max_personas is not None:
        config["max_personas"] = args.max_personas
    if args.react_mode is not None:
        config["react_mode"] = args.react_mode

    index_dir = config["index_dir"]
    output_dir = args.output_dir or config.get("output_dir", index_dir + "_react_eval")

    emb_cfg = config.get("embedding", {})
    react_reasoner_config = config.get("react_reasoner", {})
    answer_reasoner_config = config.get("answer_reasoner", {})
    react_config = dict(config.get("react", {}))
    react_mode = config.get(
        "react_mode", react_config.get("react_mode", "adaptive_react")
    )
    react_config["react_mode"] = react_mode
    if react_mode == "one_search_parity":
        parity_config_path = config.get("parity_baseline_config")
        if not parity_config_path:
            raise ValueError(
                "one_search_parity requires parity_baseline_config"
            )
        parity_config = load_config(parity_config_path)
        parity_index_dir = os.path.abspath(parity_config["index_dir"])
        if parity_index_dir != os.path.abspath(index_dir):
            raise ValueError(
                "Parity baseline and ReAct config reference different indexes: "
                f"{parity_index_dir} != {os.path.abspath(index_dir)}"
            )
        emb_cfg = dict(parity_config.get("embedding", emb_cfg))
        answer_reasoner_config = dict(parity_config.get("reasoner", {}))
        parity_retrieval = dict(parity_config.get("retrieval", {}))
        react_config.update(parity_retrieval)
        react_config["search_top_k"] = parity_retrieval.get(
            "top_k", react_config.get("search_top_k", 20)
        )
        query_policy = parity_retrieval.get("query_policy", "fixed_rewrite")
        if query_policy not in {"original", "fixed_rewrite"}:
            raise ValueError(
                "one_search_parity requires query_policy=original or fixed_rewrite"
            )
        query_manifest_path = parity_retrieval.get("query_manifest")
        if query_policy == "fixed_rewrite" and not query_manifest_path:
            query_manifest_path = os.path.join(
                PROJECT_ROOT, "manifests", "fixed_rewrite_queries.json"
            )
        rewrite_config = (
            parity_retrieval.get("rewrite_reasoner")
            or answer_reasoner_config
        )
        query_cache = QueryManifestCache(
            query_manifest_path,
            policy=query_policy,
            rewrite_model=(
                rewrite_config.get("llm_model")
                if query_policy == "fixed_rewrite" else None
            ),
        )
    elif react_mode == "adaptive_react":
        query_policy = "adaptive_react"
        query_cache = QueryManifestCache(None, policy=query_policy)
    else:
        raise ValueError(f"Unknown react_mode: {react_mode}")

    canonical_records = load_canonical_records(config.get("canonical_data_dir"))
    canonical_by_persona = records_by_persona(canonical_records)
    query_workers = args.query_workers or config.get("query_workers", 10)

    # Logging
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<level>{level: <8}</level> | {message}",
    )
    os.makedirs(output_dir, exist_ok=True)
    logger.add(
        os.path.join(output_dir, "eval.log"), level="DEBUG",
        format="{time:HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    # Load manifest
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Build the index first with build_index_*.py")
        sys.exit(1)

    with open(manifest_path) as f:
        mf = json.load(f)

    persona_entries = list(mf.get("personas", []))
    indexed_personas = {entry["persona_uuid"] for entry in persona_entries}
    for persona_uuid, records in canonical_by_persona.items():
        if persona_uuid not in indexed_personas:
            persona_entries.append({
                "persona_uuid": persona_uuid,
                "persona_name": records[0].get("persona_name", "unknown"),
                "path": os.path.join(index_dir, persona_uuid),
            })
    is_mm = _is_mm_index(mf)

    logger.info(f"Loaded manifest: {len(persona_entries)} personas")
    logger.info(f"Index type: {'Multimodal (Omni-MM)' if is_mm else 'Text (Caption→Text / Omni→Text)'}")

    if args.persona_filter:
        filters = [f.strip() for f in args.persona_filter.split(",")]
        persona_entries = [
            pe for pe in persona_entries
            if any(filt in pe["persona_uuid"] for filt in filters)
        ]
        logger.info(f"Persona filter: {len(persona_entries)} personas remaining")

    max_personas = config.get("max_personas")
    if max_personas and max_personas < len(persona_entries):
        persona_entries = persona_entries[:max_personas]
        logger.info(f"Limited to {max_personas} personas")

    selected_personas = {entry["persona_uuid"] for entry in persona_entries}
    num_selected_personas = len(selected_personas)
    expected_records = [
        record for record in canonical_records
        if record["persona_uuid"] in selected_personas
    ]
    expected_qids = [record["question_id"] for record in expected_records]

    # --- Resume logic ---
    all_entries = None
    persona_failed: dict = {}
    if args.resume:
        details_path = os.path.join(output_dir, "details.jsonl")
        if not os.path.exists(details_path):
            logger.error(f"Resume requested but no details.jsonl found: {details_path}")
            sys.exit(1)
        all_entries, persona_failed = load_failed_entries(details_path)
        total_failed = sum(len(qids) for qids in persona_failed.values())
        if total_failed == 0:
            logger.success("No failed entries found — nothing to resume.")
            return
        logger.info(
            f"Resume: {total_failed} failed entries across "
            f"{len(persona_failed)} personas"
        )
        persona_entries = [
            pe for pe in persona_entries
            if pe["persona_uuid"] in persona_failed
        ]
        logger.info(f"  Personas to re-evaluate: {len(persona_entries)}")

    # Pre-load embedding model
    if is_mm:
        logger.info(f"Loading jina omni model: {emb_cfg.get('model_name', 'jinaai/jina-embeddings-v5-omni-small')} ...")
        get_omni_embed_model(emb_cfg)
    else:
        emb_model = emb_cfg.get("model_name", "Qwen/Qwen3-Embedding-0.6B")
        emb_device = emb_cfg.get("device", "cpu")
        model_kwargs = emb_cfg.get("model_kwargs")
        processor_kwargs = emb_cfg.get("processor_kwargs")
        logger.info(f"Loading text embedding model: {emb_model} ...")
        get_text_embedding_model(emb_model, emb_device, model_kwargs, processor_kwargs)
    logger.info("Model loaded.")

    logger.info("=" * 60)
    logger.info("ReAct-RAG Evaluation")
    if args.resume:
        logger.info("  Mode:           RESUME (failed entries only)")
    logger.info(f"  Index dir:     {index_dir}")
    logger.info(f"  Index type:    {'Multimodal' if is_mm else 'Text'}")
    logger.info(f"  Output dir:    {output_dir}")
    logger.info(f"  Personas:      {len(persona_entries)}")
    logger.info(f"  Workers:       {query_workers}")
    logger.info(f"  React reasoner:{react_reasoner_config.get('llm_model', '?')}")
    logger.info(f"  Answer reasoner:{answer_reasoner_config.get('llm_model', '?')}")
    logger.info(f"  ReAct mode:     {react_mode}")
    logger.info(f"  Query policy:   {query_policy}")
    logger.info(f"  Max iterations:{react_config.get('max_iterations', 5)}")
    logger.info(f"  Search top-k:  {react_config.get('search_top_k', 10)}")
    logger.info("=" * 60)

    all_results: list[dict] = []
    failed_personas: list[dict] = []
    t_start = time.time()

    if query_workers > 1:
        with ThreadPoolExecutor(max_workers=query_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_persona_react, pe, index_dir, is_mm,
                    emb_cfg, react_reasoner_config, answer_reasoner_config, react_config,
                    canonical_by_persona.get(pe["persona_uuid"], []),
                    query_cache,
                    persona_failed.get(pe["persona_uuid"]) if args.resume else None,
                ): pe
                for pe in persona_entries
            }
            pbar = tqdm(as_completed(future_map), total=len(persona_entries),
                        desc="  Personas", unit="persona")
            for future in pbar:
                pe = future_map[future]
                try:
                    results = future.result()
                except Exception as e:
                    failed_personas.append({
                        "persona_uuid": pe["persona_uuid"], "error": str(e),
                    })
                    logger.error(f"Eval failed: {pe['persona_uuid']}: {e}")
                    all_results.extend(
                        _terminal_error_record(
                            record["item"], pe["persona_uuid"], e,
                            react_mode=react_mode,
                        )
                        for record in canonical_by_persona.get(pe["persona_uuid"], [])
                        if not args.resume
                        or record["question_id"] in persona_failed.get(
                            pe["persona_uuid"], set()
                        )
                    )
                    continue
                all_results.extend(results)
                pbar.set_postfix_str(f"{pe['persona_name'][:20]} ({len(results)} items)")
    else:
        for pe in tqdm(persona_entries, desc="  Personas", unit="persona"):
            try:
                results = evaluate_persona_react(
                    pe, index_dir, is_mm, emb_cfg,
                    react_reasoner_config, answer_reasoner_config, react_config,
                    canonical_by_persona.get(pe["persona_uuid"], []),
                    query_cache,
                    persona_failed.get(pe["persona_uuid"]) if args.resume else None,
                )
                all_results.extend(results)
            except Exception as e:
                failed_personas.append({
                    "persona_uuid": pe["persona_uuid"], "error": str(e),
                })
                logger.error(f"Eval failed: {pe['persona_uuid']}: {e}")
                all_results.extend(
                    _terminal_error_record(
                        record["item"], pe["persona_uuid"], e,
                        react_mode=react_mode,
                    )
                    for record in canonical_by_persona.get(pe["persona_uuid"], [])
                    if not args.resume
                    or record["question_id"] in persona_failed.get(
                        pe["persona_uuid"], set()
                    )
                )

    total_elapsed = time.time() - t_start
    successful = [r for r in all_results if r.get("api_status") == "ok"]

    if args.resume and all_entries is not None:
        new_by_qid = {result["question_id"]: result for result in all_results}
        patched = 0
        for entry in all_entries:
            question_id = entry.get("question_id", "")
            if question_id in new_by_qid:
                replacement = new_by_qid[question_id]
                replacement.pop("_line_idx", None)
                entry.clear()
                entry.update(replacement)
                patched += 1
        logger.info(f"Patched {patched} entries back into details.jsonl")
        for entry in all_entries:
            entry.pop("_line_idx", None)
        successful = [
            result for result in all_entries if result.get("api_status") == "ok"
        ]
        failed_items = [
            result for result in all_entries if result.get("api_status") != "ok"
        ]
    else:
        failed_items = [
            result for result in all_results if result.get("api_status") != "ok"
        ]

    result_rows = all_entries if args.resume and all_entries is not None else all_results
    qid_order = {qid: index for index, qid in enumerate(expected_qids)}
    result_rows.sort(
        key=lambda row: qid_order.get(row.get("question_id", ""), len(qid_order))
    )
    validate_result_qids(result_rows, expected_qids)

    gm = aggregate_results(successful)
    gm["retrieval"] = retrieval_metric_aggregate(result_rows)
    completed = [
        result for result in result_rows if result.get("api_status") == "ok"
    ]
    events = [
        event
        for result in result_rows
        for event in result.get("react_trace", [])
    ]
    search_events = [
        event for event in events
        if event.get("action") == "search" and event.get("status") == "ok"
    ]
    malformed_events = [
        event for event in events
        if str(event.get("action", "")).startswith("malformed")
    ]
    forced_items = sum(
        bool(result.get("trace", {}).get("answer", {}).get("forced"))
        for result in completed
    )
    refusal_items = sum(
        str(result.get("predicted_answer", "")).strip().casefold()
        in {"i don't know", "i don't know."}
        for result in completed
    )
    react_metrics = {
        "completed_items": len(completed),
        "api_failure_items": len(result_rows) - len(completed),
        "forced_answer_rate": round(forced_items / len(completed), 4) if completed else 0.0,
        "refusal_rate": round(refusal_items / len(completed), 4) if completed else 0.0,
        "searches_per_completed_item": round(
            len(search_events) / len(completed), 4
        ) if completed else 0.0,
        "unique_retrieved_per_completed_item": round(
            sum(result.get("num_retrieved", 0) for result in completed)
            / len(completed),
            4,
        ) if completed else 0.0,
        "malformed_response_events": len(malformed_events),
    }

    summary = {
        "index_dir":       index_dir,
        "output_dir":      output_dir,
        "index_type":      "multimodal" if is_mm else "text",
        "evaluated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "react_reasoner":   react_reasoner_config.get("llm_model", "?"),
        "answer_reasoner":  answer_reasoner_config.get("llm_model", "?"),
        "react_mode":      react_mode,
        "query_policy":    query_policy,
        "num_personas":    num_selected_personas,
        "num_personas_failed": len(failed_personas),
        "total_items":     len(result_rows),
        "successful_items": len(successful),
        "failed_items":    len(failed_items),
        "elapsed_seconds": round(total_elapsed, 1),
        "global_metrics":  gm,
        "react_metrics":   react_metrics,
        "failed_personas": failed_personas,
    }

    save_results(summary, result_rows, output_dir)
    index_name = os.path.basename(os.path.normpath(index_dir)).lower()
    representation = config.get("representation") or (
        "native_mm" if is_mm
        else "omni_text" if "omni_text" in index_name
        else "caption_text"
    )
    tool_rewrite = bool(
        react_config.get("tool_query_rewrite", react_config.get("query_rewrite", False))
    )
    manifest_query_policy = (
        "adaptive_react+fixed_rewrite"
        if react_mode == "adaptive_react" and tool_rewrite
        else query_policy
    )
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get(
            "canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"
        ),
        num_requested_items=len(expected_qids),
        evaluated_item_count=len(result_rows),
        execution_track=(
            "parity_validation" if react_mode == "one_search_parity"
            else "native_end_to_end"
        ),
        memory_organization="adaptive_retrieval",
        representation=representation,
        index_manifest=manifest_path,
        retriever=emb_cfg.get("model_name"),
        native_system_reader=(
            react_reasoner_config.get("llm_model")
            if react_mode == "adaptive_react" else None
        ),
        chunk_unit="session",
        query_policy=manifest_query_policy,
        answer_reasoner=(
            answer_reasoner_config.get("llm_model")
            if react_mode == "one_search_parity"
            else react_reasoner_config.get("llm_model")
        ),
        judge=config.get("judge"),
        top_k=react_config.get("search_top_k", 10),
        delivery_budget={
            key: react_config.get(key)
            for key in (
                "delivery_mode", "max_context_chars", "max_media_size_mb",
                "max_images_per_obs", "max_audio_per_obs", "max_video_per_obs",
                "max_images_per_query", "max_audio_per_query", "max_video_per_query",
                "observation_max_results", "observation_max_chars",
            )
            if react_config.get(key) is not None
        },
        parity_baseline_config=(
            config.get("parity_baseline_config")
            if react_mode == "one_search_parity" else None
        ),
    )
    print_console_summary(summary)

    logger.success(
        f"Done: {len(successful)}/{len(result_rows)} items "
        f"in {total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
