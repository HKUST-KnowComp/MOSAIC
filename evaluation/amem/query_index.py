#!/usr/bin/env python3
"""**Phase 2** — Query pre-built A-MEM indexes and evaluate retrieval.

Loads the persistent indexes created by ``build_index.py``, runs every QA
item against its persona's A-MEM instance, and computes retrieval metrics.

The QA-evaluation LLM config can be changed independently of the build phase,
so you can test different judge models without re-indexing::

    conda activate a-mem

    # Query with default config
    python evaluation/amem/query_index.py

    # Override the QA judge model
    python evaluation/amem/query_index.py --config evaluation/amem/query_config.json

    # Single-persona test
    python evaluation/amem/query_index.py --persona-filter 0083bb89 --query-workers 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path wrangling
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (
    load_config,
    build_amem_instance,
    build_evidence_map,
    compute_retrieval_metrics,
    evaluate_qa,
    aggregate_results,
    save_results,
    print_console_summary,
    load_memories_from_dicts,
)
from evaluation.protocol import (
    DeliveryPolicy,
    compute_retrieval_recall,
    load_canonical_records,
    make_item_trace,
    overlay_persona_items,
    pack_evidence,
    records_by_persona,
    validate_result_qids,
    write_run_manifest,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Load one persona from disk
# ---------------------------------------------------------------------------


def load_persona_context(persona_entry: dict, amem_config: dict) -> dict | None:
    """Reconstruct A-MEM + metadata for a single persona from disk.

    Returns a *context* dict with ``system``, ``note_registry``, ``items``,
    and ``persona``, or ``None`` on failure.
    """
    index_path = persona_entry["path"]
    persona_uuid = persona_entry["persona_uuid"]

    try:
        with open(os.path.join(index_path, "memories.json")) as f:
            mem_dicts = json.load(f)
        with open(os.path.join(index_path, "note_registry.json")) as f:
            note_registry = json.load(f)
        with open(os.path.join(index_path, "items.json")) as f:
            items = json.load(f)
    except Exception:
        logger.exception(f"Failed to load saved state for {persona_uuid}")
        return None

    # Re-open persistent ChromaDB (extend=True reconnects to existing collection)
    system = build_amem_instance(amem_config, persist_dir=index_path, extend=True)
    load_memories_from_dicts(system, mem_dicts)

    logger.debug(f"[load] {persona_entry['persona_name']}: "
                 f"{len(system.memories)} memories, {len(items)} items")

    return {
        "persona_name":  persona_entry["persona_name"],
        "persona_uuid":  persona_uuid,
        "system":        system,
        "note_registry": note_registry,
        "items":         items,
    }


def _normalize(text: str) -> str:
    """Normalize answer text for comparison."""
    text = text.strip().lower()
    for prefix in ["answer:", "the answer is:", "response:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return " ".join(text.split())


def _terminal_error_record(item: dict, persona_uuid: str, error: Exception | str) -> dict:
    message = str(error)
    question_id = item.get("question_id", "unknown")
    return {
        "question_id": question_id,
        "item_id": question_id,
        "persona_uuid": persona_uuid,
        "question_type": item.get("question_type", "unknown"),
        "answerability": item.get("answerability", "unknown"),
        "question": item.get("question", ""),
        "predicted_answer": "",
        "api_status": "error",
        "evaluation": {
            "exact_match": False,
            "alias_match": False,
            "unanswerable_correct": False,
        },
        "error": message,
        "trace": make_item_trace(
            question_id=question_id,
            query={
                "original": item.get("question", ""),
                "retrieval_query": item.get("question", ""),
                "rewrite_model": None,
            },
            evidence_plan=item.get("evidence_plan", []),
            answer={
                "predicted": "",
                "api_status": "error",
                "forced": False,
                "error_class": type(error).__name__ if isinstance(error, Exception) else "Error",
                "error_message": message,
            },
            required_session_ids=item.get("answer_session_ids", []),
        ),
    }


# ---------------------------------------------------------------------------
# Query one item
# ---------------------------------------------------------------------------


def query_single_item(
    item: dict,
    persona_uuid: str,
    system,
    note_registry: dict,
    k_values: list[int],
    qa_config: dict | None,
    persona_lock: threading.Lock,
) -> dict:
    """Run one QA item against its persona's A-MEM (thread-safe)."""
    item_id = item.get("question_id", "unknown")
    question = item.get("question", "")
    question_type = item.get("question_type", "unknown")
    answerability = item.get("answerability", "unknown")
    max_k = max(k_values)

    try:
        with persona_lock:
            search_results = system.search_agentic(question, k=max_k)

        evidence_map = build_evidence_map(item, note_registry)

        metrics_by_k = {}
        for k in k_values:
            metrics_by_k[f"@{k}"] = compute_retrieval_metrics(
                search_results, evidence_map["evidence_note_ids"], k,
            )

        retrieved = []
        for rank, result in enumerate(search_results[:max_k], start=1):
            note_id = result["id"]
            meta = note_registry.get(note_id, {})
            retrieved.append({
                "rank": rank,
                "chunk_id": note_id,
                "session_id": meta.get("session_id", ""),
                "timestamp": meta.get("timestamp", ""),
                "content": result.get("content", ""),
                "score": result.get("score"),
                "modality": "text",
                "fact_key": meta.get("fact_key"),
                "has_answer": meta.get("has_answer"),
            })
        packed = pack_evidence(
            retrieved,
            DeliveryPolicy(
                mode="production_budget",
                attach_media=False,
                max_context_chars=(qa_config or {}).get("max_context_chars"),
            ),
        )

        # ---- run LLM QA evaluation if enabled ----
        qa_result = None
        predicted_answer = ""
        if qa_config and qa_config.get("enabled"):
            qa_result = evaluate_qa(
                item, packed["delivered"], qa_config.get("llm_config", {}),
            )
            if qa_result and "predicted_answer" in qa_result:
                predicted_answer = qa_result["predicted_answer"]

        # ---- string-match evaluation (for compatibility with judge) ----
        expected = item.get("answer", "")
        aliases = item.get("answer_aliases", [])
        norm_pred = _normalize(predicted_answer)
        norm_exp = _normalize(expected)
        norm_aliases = [_normalize(a) for a in aliases]
        exact_match = norm_pred == norm_exp
        alias_match = exact_match or any(norm_pred == a for a in norm_aliases)
        unanswerable_correct = (
            answerability == "unanswerable"
            and (
                "unanswerable" in predicted_answer.lower()
                or "i don't know" in predicted_answer.lower()
            )
        )

        retrieval_recall = compute_retrieval_recall(
            retrieved,
            item.get("evidence_plan", []),
            k_values,
            required_session_ids=item.get("answer_session_ids", []),
        )
        api_status = (
            "error" if qa_result and qa_result.get("error")
            else "empty" if not predicted_answer.strip()
            else "ok"
        )
        trace = make_item_trace(
            question_id=item_id,
            query={
                "original": question,
                "retrieval_query": question,
                "rewrite_model": None,
            },
            ranked=retrieved,
            evidence_plan=item.get("evidence_plan", []),
            delivery=packed,
            answer={
                "predicted": predicted_answer,
                "api_status": api_status,
                "forced": False,
                **({"error_message": qa_result["error"]} if qa_result and qa_result.get("error") else {}),
            },
            required_session_ids=item.get("answer_session_ids", []),
        )

        result = {
            "question_id":          item_id,
            "item_id":              item_id,
            "persona_uuid":         persona_uuid,
            "question_type":        question_type,
            "answerability":        answerability,
            "question":             question,
            "expected_answer":      expected,
            "answer_aliases":       aliases,
            "predicted_answer":     predicted_answer,
            "api_status":           api_status,
            "chain_description":    item.get("chain_description", ""),
            "multihop_chain":       item.get("multihop_chain", ""),
            "evidence_plan": [
                {
                    k: ev[k] for k in (
                        "fact_statement", "role", "modality",
                        "required_for_answer", "modality_rationale",
                    ) if k in ev
                }
                for ev in item.get("evidence_plan", [])
            ],
            "evaluation": {
                "exact_match":       exact_match,
                "alias_match":       alias_match,
                "unanswerable_correct": unanswerable_correct,
            },
            "retrieval_recall":     retrieval_recall,
            "trace":                trace,
            "retrieved":            retrieved,
            "memory_read_result": {
                "backend": "a_mem",
                "stored_representation": "text_evolving_notes",
                "retrieved_units": retrieved,
                "delivered_units": packed["delivered"],
                "backend_native_answer": predicted_answer,
                "observability": {
                    "write_provenance": True,
                    "retrieval_ranks": True,
                    "delivery_manifest": True,
                },
            },
            "metrics":              metrics_by_k,
            "num_notes_in_pool":    len(note_registry),
            "num_evidence_notes":   evidence_map["num_evidence"],
            "evidence_note_ids":    sorted(evidence_map["evidence_note_ids"]),
            "retrieved_note_ids":   [r["id"] for r in search_results[:max_k]],
            "qa_result":            qa_result,
        }
        if api_status == "error":
            result["error"] = qa_result.get("error", "QA call failed")
        return result

    except Exception as error:
        logger.error(f"Query failed for {item_id}: {error}")
        return {
            "question_id": item_id,
            "item_id":       item_id,
            "persona_uuid":  persona_uuid,
            "question_type": question_type,
            "answerability": answerability,
            "question":      question,
            "predicted_answer": "",
            "api_status": "error",
            "evaluation": {
                "exact_match": False,
                "alias_match": False,
                "unanswerable_correct": False,
            },
            "error":         str(error),
            "trace": make_item_trace(
                question_id=item_id,
                query={"original": question, "retrieval_query": question, "rewrite_model": None},
                evidence_plan=item.get("evidence_plan", []),
                answer={
                    "predicted": "",
                    "api_status": "error",
                    "forced": False,
                    "error_class": type(error).__name__,
                    "error_message": str(error),
                },
                required_session_ids=item.get("answer_session_ids", []),
            ),
        }


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query pre-built A-MEM indexes and evaluate retrieval"
    )
    p.add_argument(
        "--config", default=os.path.join(HERE, "query_config.json"),
        help="Path to query config JSON",
    )
    p.add_argument(
        "--index-dir", default=None,
        help="Override index directory (default from config)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Override output directory for results",
    )
    p.add_argument(
        "--max-personas", type=int, default=None,
        help="Limit personas to query",
    )
    p.add_argument(
        "--persona-filter", default=None,
        help="Comma-separated persona UUID substrings",
    )
    p.add_argument(
        "--query-workers", type=int, default=None,
        help="Parallel workers for queries (default from config, or 1)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.max_personas is not None:
        config["max_personas"] = args.max_personas
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    index_dir = args.index_dir or config.get("index_dir", config["output_dir"])
    config["index_dir"] = index_dir
    canonical_records = load_canonical_records(config.get("canonical_data_dir"))
    canonical_by_persona = records_by_persona(canonical_records)

    # Logging — ERROR+ to stderr (clean terminal), INFO+ to file
    logger.remove()
    log_level = config.get("logging", {}).get("level", "INFO")
    logger.add(
        sys.stderr, level="ERROR", colorize=True,
        format="<level>{level: <8}</level> | {message}",
    )
    log_file = config.get("logging", {}).get("file")
    if not log_file:
        log_file = os.path.join(config["output_dir"], "query.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logger.add(log_file, level=log_level)
    logger.info(f"Log file: {log_file}")

    # Load manifest
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found: {manifest_path}")
        logger.error("Run build_index.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    personas = manifest.get("personas", [])
    if not personas:
        logger.error("Manifest contains no personas — run build_index.py first.")
        sys.exit(1)

    logger.info(f"Loaded manifest: {len(personas)} personas, "
                f"{manifest.get('total_notes', '?')} notes "
                f"(built {manifest.get('built_at', '?')})")

    # Filter
    if args.persona_filter:
        filters = [f.strip() for f in args.persona_filter.split(",")]
        personas = [p for p in personas
                    if any(filt in p.get("persona_uuid", "") for filt in filters)]
        logger.info(f"Persona filter: {len(personas)} personas remaining")

    max_personas = config.get("max_personas")
    if max_personas and max_personas < len(personas):
        personas = personas[:max_personas]
        logger.info(f"Limited to {len(personas)} personas")

    selected_personas = {entry["persona_uuid"] for entry in personas}
    expected_records = [
        record for record in canonical_records
        if record["persona_uuid"] in selected_personas
    ]
    expected_qids = [record["question_id"] for record in expected_records]

    amem_config = config.get("amem_config", {})
    k_values = config.get("k_values", [1, 5, 10, 20])
    qa_config = config.get("qa_evaluation")
    query_workers = args.query_workers or config.get("query_workers", 1)

    logger.info("=" * 60)
    logger.info("A-MEM Query Runner")
    logger.info(f"  Index dir:     {index_dir}")
    logger.info(f"  Output dir:    {config['output_dir']}")
    logger.info(f"  Personas:      {len(personas)}")
    logger.info(f"  k values:      {k_values}")
    logger.info(f"  QA eval:       {qa_config.get('enabled', False) if qa_config else False}")
    logger.info(f"  Query workers: {query_workers}")
    logger.info("=" * 60)

    # ---- Load all personas -------------------------------------------------
    logger.info("Loading indexes ...")
    t_load = time.time()
    persona_contexts: list[dict] = []
    failed_loads = 0
    load_error_results: list[dict] = []

    for entry in tqdm(personas, desc="  Load", unit="persona"):
        ctx = load_persona_context(entry, amem_config)
        if ctx is None:
            failed_loads += 1
            load_error_results.extend(
                _terminal_error_record(
                    record["item"], entry["persona_uuid"], "Failed to load A-MEM store"
                )
                for record in canonical_by_persona.get(entry["persona_uuid"], [])
            )
            continue
        ctx["items"] = overlay_persona_items(
            ctx["items"], canonical_by_persona.get(entry["persona_uuid"], [])
        )
        persona_contexts.append(ctx)

    load_time = time.time() - t_load
    logger.success(
        f"Loaded {len(persona_contexts)}/{len(personas)} personas "
        f"in {load_time:.1f}s"
    )

    # ---- Query all items ---------------------------------------------------
    logger.info("Querying ...")
    t_query = time.time()

    all_work: list[tuple[dict, str, Any, dict, threading.Lock]] = []
    for ctx in persona_contexts:
        lock = threading.Lock()
        for item in ctx["items"]:
            all_work.append((
                item,
                ctx["persona_uuid"],
                ctx["system"],
                ctx["note_registry"],
                lock,
            ))

    logger.info(f"  {len(all_work)} total queries")
    all_item_results: list[dict] = list(load_error_results)
    failed_queries = len(load_error_results)

    if query_workers > 1:
        with ThreadPoolExecutor(max_workers=query_workers) as executor:
            futures = {
                executor.submit(
                    query_single_item,
                    item, persona_uuid, system, registry, k_values, qa_config, lock,
                ): (item, persona_uuid)
                for item, persona_uuid, system, registry, lock in all_work
            }
            pbar = tqdm(as_completed(futures), total=len(futures),
                        desc="  Query", unit="item")
            for future in pbar:
                try:
                    result = future.result()
                except Exception as error:
                    failed_queries += 1
                    item, persona_uuid = futures[future]
                    result = _terminal_error_record(item, persona_uuid, error)
                if "error" in result:
                    failed_queries += 1
                all_item_results.append(result)
    else:
        for item, persona_uuid, system, registry, lock in tqdm(all_work, desc="  Query", unit="item"):
            result = query_single_item(
                item, persona_uuid, system, registry, k_values, qa_config, lock
            )
            if "error" in result:
                failed_queries += 1
            all_item_results.append(result)

    query_time = time.time() - t_query
    failed_queries = sum(
        result.get("api_status") != "ok" for result in all_item_results
    )
    successful = len(all_item_results) - failed_queries
    logger.success(f"Queried {successful}/{len(all_work)} items in {query_time:.1f}s")

    # ---- Aggregate & save --------------------------------------------------
    summary = aggregate_results(all_item_results, k_values)
    summary["num_personas"] = len(personas)
    summary["num_personas_loaded"] = len(persona_contexts)
    summary["num_personas_failed"] = failed_loads
    summary["total_items"] = len(all_item_results)
    summary["successful_items"] = successful
    summary["failed_items"] = failed_queries
    summary["load_seconds"] = round(load_time, 1)
    summary["query_seconds"] = round(query_time, 1)
    summary["index_dir"] = index_dir

    qid_order = {qid: index for index, qid in enumerate(expected_qids)}
    all_item_results.sort(
        key=lambda row: qid_order.get(row.get("question_id", ""), len(qid_order))
    )
    validate_result_qids(all_item_results, expected_qids)

    save_results(summary, all_item_results, config["output_dir"])
    index_name = os.path.basename(os.path.normpath(index_dir)).lower()
    representation = "omni_text" if "omni" in index_name else "caption_text"
    qa_llm = (qa_config or {}).get("llm_config", {})
    write_run_manifest(
        config["output_dir"],
        config,
        canonical_manifest=config.get("canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"),
        num_requested_items=len(expected_qids),
        evaluated_item_count=len(all_item_results),
        execution_track="native_end_to_end",
        memory_organization="text_symbolic",
        representation=representation,
        index_manifest=manifest_path,
        retriever="A-MEM search_agentic",
        native_system_reader=qa_llm.get("llm_model"),
        chunk_unit="memory_note",
        query_policy="original",
        answer_reasoner=qa_llm.get("llm_model"),
        judge=config.get("judge"),
        top_k=max(k_values),
        delivery_budget={"mode": "production_budget", "media": "text_only"},
    )
    print_console_summary(summary)


if __name__ == "__main__":
    main()
