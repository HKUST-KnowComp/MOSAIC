#!/usr/bin/env python3
"""**RAG-Text** — Evaluate text-based RAG on Caption→Text or Omni→Text indexes.

Loads pre-built text FAISS indexes, retrieves relevant context for each
question, and answers using a configurable text LLM reasoner.

Usage::

    # Evaluate caption→text index
    python evaluation/rag/eval_rag_text.py

    # Evaluate omni→text index
    python evaluation/rag/eval_rag_text.py \
        --config evaluation/rag/config_eval_omni_text_deepseekv4flash.json

    # Smoke-test: 2 personas
    python evaluation/rag/eval_rag_text.py --max-personas 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    encode_text_batch,
    load_faiss_index,
    search_faiss_index_filtered,
    match_answer,
    compute_retrieval_recall,
    rewrite_query,
    aggregate_results,
    partition_terminal_results,
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
    overlay_persona_items,
    pack_evidence,
    records_by_persona,
    validate_result_qids,
    write_run_manifest,
)

PROJECT_ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# Time-cutoff filter
# ---------------------------------------------------------------------------

FALLBACK_DATE_FMT = "%Y/%m/%d"


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string into a datetime. Returns None on failure."""
    if not date_str:
        return None
    for fmt in [FALLBACK_DATE_FMT, "%Y-%m-%d", "%Y/%m/%d (%a) %H:%M"]:
        try:
            return datetime.strptime(date_str.split(" (")[0], FALLBACK_DATE_FMT)
        except (ValueError, TypeError):
            continue
    return None


def filter_by_time(chunks: list[dict], question_date: str) -> list[dict]:
    """Filter chunks to those with timestamp <= question_date."""
    q_date = _parse_date(question_date)
    if q_date is None:
        return chunks  # can't filter, return all

    filtered = []
    for c in chunks:
        c_ts = c.get("timestamp", "")
        c_date = _parse_date(c_ts)
        if c_date is None or c_date <= q_date:
            filtered.append(c)

    return filtered


# ---------------------------------------------------------------------------
# LLM reasoner (OpenAI-compatible)
# ---------------------------------------------------------------------------


def call_text_reasoner(
    question: str,
    context_chunks: list[dict],
    reasoner_config: dict,
    question_type: str | None = None,
) -> str:
    """Call a text LLM to answer a question given retrieved context.

    Returns the predicted answer string.
    """
    from _common import evidence_update_answer_rule

    # Build context from chunks
    context_parts = []
    for i, c in enumerate(context_chunks):
        context_parts.append(
            f"[{i + 1}] score={c.get('score', 0):.3f} | "
            f"{c.get('timestamp', '')} | {c.get('modality', 'text')}\n"
            f"    {c.get('content', '')}"
        )
    context_str = "\n\n".join(context_parts)

    system_prompt = (
        "You are an evaluation assistant. Your task is to answer questions "
        "about a user based on retrieved conversation history.\n\n"
        "Answer rules:\n"
        "1. Answer using ONLY the retrieved evidence provided. Do not use "
        "outside knowledge.\n"
        "2. Your answer should be as short as possible — typically a single "
        "phrase, number, name, or date.\n"
        "3. Do not include reasoning, explanations, or hedging in your final "
        "answer. Extract the requested fact directly.\n"
        "4. If the retrieved evidence does not contain enough information to "
        'answer the question, respond with exactly: I don\'t know.\n'
        f"5. {evidence_update_answer_rule(question_type)}\n"
        '6. For temporal references in the question ("currently", "now", '
        '"recently"), interpret them relative to the question\'s date.\n'
        "7. Examples of correct answer format:\n"
        '   - Question: "What was the work start time?" → Answer: "8:15 AM"\n'
        '   - Question: "What did the charity race raise awareness for?" → Answer: "mental health"\n'
        '   - Question: "How many books did I read this year?" → Answer: "12"\n'
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

    try:
        client = create_openai_client(reasoner_config)

        kwargs = {
            "model": reasoner_config.get("llm_model", "gpt-4o"),
            "messages": messages,
            "temperature": reasoner_config.get("temperature", 0.0),
            "max_tokens": reasoner_config.get("max_tokens", 200),
        }

        _apply_extra_body(kwargs, reasoner_config)

        logger.debug(f"[Reasoner Prompt]\nsystem: {system_prompt}\n\nuser: {user_prompt}")

        answer = call_llm_with_retry(
            client,
            kwargs,
            max_retries=int(reasoner_config.get("max_retries", 2)),
            backoff=float(reasoner_config.get("retry_backoff", 2.0)),
        )

        logger.debug(f"[Reasoner Output] {answer}")
        return answer

    except Exception as e:
        logger.warning(f"Reasoner call failed: {e}")
        return "ERROR: " + str(e)


# ---------------------------------------------------------------------------
# Per-persona evaluation
# ---------------------------------------------------------------------------


def _terminal_error_record(
    item: dict,
    persona_uuid: str,
    error: Exception | str,
    query_info: dict | None = None,
) -> dict:
    message = str(error)
    question_id = item.get("question_id", "unknown")
    query_info = query_info or {
        "original": item.get("question", ""),
        "retrieval_query": item.get("question", ""),
        "rewrite_model": None,
    }
    trace = make_item_trace(
        question_id=question_id,
        query=query_info,
        evidence_plan=item.get("evidence_plan", []),
        answer={
            "predicted": "",
            "api_status": "error",
            "forced": False,
            "error_class": type(error).__name__ if isinstance(error, Exception) else "Error",
            "error_message": message,
        },
        required_session_ids=item.get("answer_session_ids", []),
    )
    return {
        "question_id": question_id,
        "question_type": item.get("question_type", "unknown"),
        "answerability": item.get("answerability", "unknown"),
        "question": item.get("question", ""),
        "expected_answer": item.get("answer", ""),
        "predicted_answer": "",
        "evaluation": {
            "exact_match": False,
            "alias_match": False,
            "unanswerable_correct": False,
        },
        "error": message,
        "api_status": "error",
        "trace": trace,
        "persona_uuid": persona_uuid,
    }


def evaluate_persona(
    persona_entry: dict,
    index_dir: str,
    embedding_model_name: str,
    embedding_device: str,
    reasoner_config: dict,
    retrieval_config: dict,
    canonical_records: list[dict],
    query_cache: QueryManifestCache,
    only_qids: set | None = None,
) -> list[dict]:
    """Load a persona's index and evaluate its questions.

    If *only_qids* is provided, only those question_ids are evaluated.
    Returns a list of per-question result dicts.
    """
    persona_uuid = persona_entry["persona_uuid"]
    persona_path = persona_entry["path"]

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
        return [_terminal_error_record(item, persona_uuid, error) for item in items]

    items = overlay_persona_items(index_items, canonical_records, only_qids=only_qids)

    top_k = retrieval_config.get("top_k", 20)
    enable_time_filter = retrieval_config.get("enable_time_filter", False)
    rewrite_reasoner = retrieval_config.get("rewrite_reasoner")  # optional separate LLM for rewrite
    score_threshold = float(retrieval_config.get("score_threshold", 0.0))
    candidate_k = int(
        retrieval_config.get(
            "candidate_k",
            top_k if not enable_time_filter and score_threshold <= 0 else max(top_k * 2, top_k),
        )
    )

    results = []
    for item in tqdm(items, desc=f"  {persona_uuid[:8]}", unit="q", leave=False):
        question_id = item.get("question_id", "unknown")
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
        }

        try:
            rewrite_cfg = rewrite_reasoner or reasoner_config
            query_info = query_cache.resolve(
                question_id,
                question,
                lambda value: rewrite_query(value, rewrite_cfg),
            )
            search_query = query_info["retrieval_query"]

            # Embed search query
            q_embedding = encode_text_batch(
                [search_query], embedding_model_name, embedding_device,
                prompt_name="query",
            )
            q_embedding = q_embedding.astype("float32")

            # Search FAISS
            def _eligible(unit: dict) -> bool:
                if enable_time_filter and not filter_by_time([unit], question_date):
                    return False
                return unit.get("score", 0) >= score_threshold

            retrieved = search_faiss_index_filtered(
                index,
                q_embedding,
                metadata,
                k=top_k,
                candidate_k=candidate_k,
                eligibility_fn=_eligible,
            )
            n_ret = len(retrieved)
            if n_ret:
                logger.debug(
                    f"[Retrieval] FAISS: {n_ret} results, "
                    f"top_score={retrieved[0]['score']:.4f}"
                )
            else:
                logger.debug(
                    f"[Retrieval] FAISS: 0 results "
                    f"(index.ntotal={index.ntotal})"
                )

            packed = pack_evidence(
                retrieved,
                DeliveryPolicy(
                    mode="production_budget",
                    attach_media=False,
                    max_context_chars=retrieval_config.get("max_context_chars"),
                ),
            )
            delivered = packed["delivered"]
            predicted = call_text_reasoner(
                question, delivered, reasoner_config,
                question_type=question_type,
            )

            api_status = (
                "error" if predicted.startswith("ERROR:")
                else "empty" if not predicted.strip()
                else "ok"
            )
            if api_status == "ok":
                eval_result = match_answer(predicted, expected, aliases, answerability)
            else:
                eval_result = {
                    "exact_match": False,
                    "alias_match": False,
                    "unanswerable_correct": False,
                }

            # Retrieval recall against evidence plan
            recall_metrics = compute_retrieval_recall(
                retrieved, item.get("evidence_plan", []),
            )

            trace = make_item_trace(
                question_id=question_id,
                query=query_info,
                ranked=retrieved,
                evidence_plan=item.get("evidence_plan", []),
                delivery=packed,
                answer={
                    "predicted": predicted,
                    "api_status": api_status,
                    "forced": False,
                },
                required_session_ids=item.get("answer_session_ids", []),
            )

            latency = time.time() - t0
            # Reasoning context from the benchmark item
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
                "trace": trace,
                "latency_seconds": round(latency, 3),
                "num_retrieved": len(retrieved),
                "retrieved": [
                    {
                        "chunk_id": r.get("chunk_id", ""),
                        "score": r.get("score", 0),
                        "modality": r.get("modality", "text"),
                        "content": (r.get("content", "") or "")[:300],
                        "session_id": r.get("session_id", ""),
                        "timestamp": r.get("timestamp", ""),
                        "fact_key": r.get("fact_key"),
                        "fact_keys": r.get("fact_keys", []),
                        "pre_filter_rank": r.get("pre_filter_rank"),
                        "eligible_rank": r.get("eligible_rank"),
                    }
                    for r in retrieved[:20]
                ],
                "persona_uuid": persona_uuid,
            })

        except Exception as error:
            logger.warning(f"Failed to evaluate {question_id}: {error}")
            results.append(
                _terminal_error_record(item, persona_uuid, error, query_info)
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RAG-Text: Evaluate text-based RAG on Caption→Text or Omni→Text indexes"
    )
    p.add_argument(
        "--config",
        default=os.path.join(HERE, "config_eval_caption_text_deepseekv4flash.json"),
        help="Path to eval config JSON",
    )
    p.add_argument(
        "--max-personas", type=int, default=None,
        help="Limit number of persona indexes to evaluate",
    )
    p.add_argument(
        "--persona-filter", default=None,
        help="Comma-separated persona UUID substrings",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override output directory (default from config)",
    )
    p.add_argument(
        "--query-workers", type=int, default=None,
        help="Parallel query workers (default from config)",
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

    index_dir = config["index_dir"]
    output_dir = args.output_dir or config.get("output_dir", index_dir + "_eval_results")

    emb_cfg = config.get("embedding", {})
    model_name = emb_cfg.get("model_name", "all-MiniLM-L6-v2")
    device = emb_cfg.get("device", "cpu")
    model_kwargs = emb_cfg.get("model_kwargs")
    processor_kwargs = emb_cfg.get("processor_kwargs")

    reasoner_config = config.get("reasoner", {})
    retrieval_config = config.get("retrieval", {"top_k": 20})
    query_workers = args.query_workers or config.get("query_workers", 100)
    query_policy = retrieval_config.get("query_policy")
    if not query_policy:
        query_policy = "fixed_rewrite" if retrieval_config.get("query_rewrite") else "original"
    query_manifest_path = retrieval_config.get("query_manifest")
    if query_policy == "fixed_rewrite" and not query_manifest_path:
        query_manifest_path = os.path.join(output_dir, "query_manifest.json")
    rewrite_config = retrieval_config.get("rewrite_reasoner") or reasoner_config
    query_cache = QueryManifestCache(
        query_manifest_path,
        policy=query_policy,
        rewrite_model=rewrite_config.get("llm_model") if query_policy == "fixed_rewrite" else None,
    )

    canonical_records = load_canonical_records(config.get("canonical_data_dir"))
    canonical_by_persona = records_by_persona(canonical_records)

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
        logger.error("Build the index first with build_index_caption_text.py or build_index_omni_text.py")
        sys.exit(1)

    with open(manifest_path) as f:
        mf = json.load(f)

    persona_entries = mf.get("personas", [])
    indexed_personas = {entry["persona_uuid"] for entry in persona_entries}
    for persona_uuid, records in canonical_by_persona.items():
        if persona_uuid not in indexed_personas:
            persona_entries.append({
                "persona_uuid": persona_uuid,
                "persona_name": records[0].get("persona_name", "unknown"),
                "path": os.path.join(index_dir, persona_uuid),
            })
    logger.info(f"Loaded manifest: {len(persona_entries)} personas")

    # Filter
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
    get_text_embedding_model(model_name, device, model_kwargs, processor_kwargs)

    logger.info("=" * 60)
    logger.info("RAG-Text: Text RAG Evaluation")
    if args.resume:
        logger.info("  Mode:           RESUME (failed entries only)")
    logger.info(f"  Index dir:     {index_dir}")
    logger.info(f"  Output dir:    {output_dir}")
    logger.info(f"  Personas:      {len(persona_entries)}")
    logger.info(f"  Workers:       {query_workers}")
    logger.info(f"  Embedding:     {model_name} on {device}")
    logger.info(f"  Reasoner:      {reasoner_config.get('llm_model', '?')}")
    logger.info(f"  Top-k:         {retrieval_config.get('top_k', 20)}")
    logger.info(f"  Time filter:   {retrieval_config.get('enable_time_filter', True)}")
    logger.info("=" * 60)

    all_results: list[dict] = []
    failed_personas: list[dict] = []
    t_start = time.time()

    if query_workers > 1:
        with ThreadPoolExecutor(max_workers=query_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_persona, pe, index_dir, model_name, device,
                    reasoner_config, retrieval_config,
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
                        _terminal_error_record(record["item"], pe["persona_uuid"], e)
                        for record in canonical_by_persona.get(pe["persona_uuid"], [])
                        if not args.resume
                        or record["question_id"] in persona_failed.get(pe["persona_uuid"], set())
                    )
                    continue
                all_results.extend(results)
                pbar.set_postfix_str(
                    f"{pe['persona_name'][:20]} "
                    f"({len(results)} items)"
                )
    else:
        for pe in tqdm(persona_entries, desc="  Personas", unit="persona"):
            try:
                results = evaluate_persona(
                    pe, index_dir, model_name, device,
                    reasoner_config, retrieval_config,
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
                    _terminal_error_record(record["item"], pe["persona_uuid"], e)
                    for record in canonical_by_persona.get(pe["persona_uuid"], [])
                    if not args.resume
                    or record["question_id"] in persona_failed.get(pe["persona_uuid"], set())
                )

    total_elapsed = time.time() - t_start
    successful, failed_items = partition_terminal_results(all_results)

    if args.resume and all_entries is not None:
        new_by_qid = {r["question_id"]: r for r in all_results}
        patched = 0
        for entry in all_entries:
            qid = entry.get("question_id", "")
            if qid in new_by_qid:
                new_result = new_by_qid[qid]
                new_result.pop("_line_idx", None)
                entry.clear()
                entry.update(new_result)
                patched += 1
        logger.info(f"Patched {patched} entries back into details.jsonl")
        details_path = os.path.join(output_dir, "details.jsonl")
        with open(details_path, "w") as f:
            for entry in all_entries:
                entry.pop("_line_idx", None)
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        successful, failed_items = partition_terminal_results(all_entries)

    result_rows = all_entries if args.resume and all_entries is not None else all_results
    qid_order = {qid: index for index, qid in enumerate(expected_qids)}
    result_rows.sort(key=lambda row: qid_order.get(row.get("question_id", ""), len(qid_order)))
    validate_result_qids(result_rows, expected_qids)

    # Compute metrics
    gm = aggregate_results(successful)

    summary = {
        "index_dir":       index_dir,
        "output_dir":      output_dir,
        "evaluated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name":      model_name,
        "reasoner_model":  reasoner_config.get("llm_model", "?"),
        "num_personas":    len(persona_entries),
        "num_personas_failed": len(failed_personas),
        "total_items":     len(all_entries if args.resume else all_results),
        "successful_items": len(successful),
        "failed_items":    len(failed_items),
        "elapsed_seconds": round(total_elapsed, 1),
        "global_metrics":  gm,
        "failed_personas": failed_personas,
    }

    save_results(summary, result_rows, output_dir)
    index_name = os.path.basename(os.path.normpath(index_dir)).lower()
    representation = config.get("representation") or (
        "omni_text" if "omni_text" in index_name else "caption_text"
    )
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get("canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"),
        num_requested_items=len(expected_qids),
        evaluated_item_count=len(result_rows),
        execution_track=config.get("execution_track", "native_end_to_end"),
        memory_organization="flat_index",
        representation=representation,
        index_manifest=manifest_path,
        retriever=model_name,
        native_system_reader=None,
        chunk_unit="session",
        query_policy=query_policy,
        answer_reasoner=reasoner_config.get("llm_model"),
        judge=config.get("judge"),
        top_k=retrieval_config.get("top_k", 20),
        delivery_budget={"mode": "production_budget", "media": "text_only"},
    )
    print_console_summary(summary)

    logger.success(
        f"Done: {len(successful)}/{len(all_results)} items "
        f"in {total_elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
