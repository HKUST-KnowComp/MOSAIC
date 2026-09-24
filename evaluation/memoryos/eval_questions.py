#!/usr/bin/env python3
"""Phase 2: Load pre-built MemoryOS memories and evaluate all QA questions.

Requires build_memories.py to have been run first (memories are loaded from disk).

Usage:
  cd /path/to/MOSAIC
  python evaluation/memoryos/eval_questions.py --config evaluation/memoryos/memoryos_eval_config.json
  python evaluation/memoryos/eval_questions.py --config ... --persona-filter 0083bb89 --eval-workers 200
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common import (
    MemoryOSEvalWrapper,
    compute_metrics,
    evaluate_answer,
    load_eval_config,
    load_json,
    save_json,
    _collect_all_items,
    _print_summary,
)
from evaluation.protocol import (
    DeliveryPolicy,
    load_canonical_records,
    make_item_trace,
    pack_evidence,
    records_by_persona,
    validate_result_qids,
    write_details_jsonl,
    write_run_manifest,
)


# ---------------------------------------------------------------------------
# Phase 2: Evaluate one question
# ---------------------------------------------------------------------------

def _memory_read_units(context: dict) -> list[dict]:
    units: list[dict] = []
    for source, rows in (
        ("short_term", context.get("short_term_history", [])),
        ("memory_page", context.get("retrieved_pages", [])),
        ("user_knowledge", context.get("retrieved_user_knowledge", [])),
        ("assistant_knowledge", context.get("retrieved_assistant_knowledge", [])),
    ):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                row = {"content": str(row)}
            content = row.get("knowledge") or "\n".join(
                str(row.get(key, ""))
                for key in ("user_input", "agent_response", "meta_info")
                if row.get(key)
            )
            units.append({
                "rank": len(units) + 1,
                "chunk_id": str(
                    row.get("page_id")
                    or row.get("id")
                    or f"{source}:{index}"
                ),
                "session_id": row.get("session_id", ""),
                "timestamp": row.get("timestamp", ""),
                "modality": "text",
                "fact_keys": [],
                "score": row.get("score", row.get("similarity")),
                "content": content,
                "memory_tier": source,
            })
    profile = context.get("user_profile")
    if profile:
        units.append({
            "rank": len(units) + 1,
            "chunk_id": "user_profile",
            "session_id": "",
            "timestamp": "",
            "modality": "text",
            "fact_keys": [],
            "score": None,
            "content": str(profile),
            "memory_tier": "profile",
        })
    return units


def _terminal_error(item: dict, persona_uuid: str, error: Exception | str) -> dict:
    message = str(error)
    question_id = item.get("question_id", "")
    trace = make_item_trace(
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
    )
    trace["retrieval"]["status"] = "unobservable"
    return {
        "question_id": question_id,
        "persona_uuid": persona_uuid,
        "question_type": item.get("question_type", "unknown"),
        "answerability": item.get("answerability", "unknown"),
        "question": item.get("question", ""),
        "ground_truth_answer": item.get("answer", ""),
        "predicted_answer": "",
        "api_status": "error",
        "evaluation": {
            "exact_match": False,
            "alias_match": False,
            "unanswerable_correct": False,
        },
        "error": message,
        "trace": trace,
    }


def _eval_one(item: dict, persona_uuid: str, wrapper: MemoryOSEvalWrapper,
              llm_cfg: dict, rcfg: dict) -> dict:
    """Evaluate a single question. Returns item result dict."""
    question = item.get("question", "")
    answerability = item.get("answerability", "answerable")
    ground_truth = item.get("answer", "")
    aliases = item.get("answer_aliases", [])

    start = time.time()
    try:
        predicted, memory_context = wrapper.query_eval(
            question,
            llm_cfg,
            rcfg,
            return_context=True,
            question_type=item.get("question_type"),
        )
    except Exception as error:
        return _terminal_error(item, persona_uuid, error)
    latency = time.time() - start

    evaluation = evaluate_answer(predicted, ground_truth, aliases)
    if answerability == "unanswerable":
        evaluation["unanswerable_correct"] = (
            "unanswerable" in predicted.lower()
            or "i don't know" in predicted.lower()
        )

    retrieved_units = _memory_read_units(memory_context)
    packed = pack_evidence(
        retrieved_units,
        DeliveryPolicy(mode="production_budget", attach_media=False),
    )
    api_status = "empty" if not predicted.strip() else "ok"
    trace = make_item_trace(
        question_id=item.get("question_id", ""),
        query={"original": question, "retrieval_query": question, "rewrite_model": None},
        ranked=retrieved_units,
        evidence_plan=item.get("evidence_plan", []),
        delivery=packed,
        answer={"predicted": predicted, "api_status": api_status, "forced": False},
        required_session_ids=item.get("answer_session_ids", []),
    )
    # MemoryOS does not preserve benchmark fact provenance in exported pages.
    # Keep the context observable while marking fact-level retrieval diagnosis
    # unavailable rather than reporting a fabricated zero.
    trace["retrieval"].update({
        "status": "unobservable",
        "fact_recall_at_k": "unobservable",
        "complete_chain_at_k": "unobservable",
        "required_session_recall_at_k": "unobservable",
    })
    trace["delivery"].update({
        "required_fact_coverage": "unobservable",
        "complete_chain_covered": "unobservable",
    })

    return {
        "question_id": item.get("question_id", ""),
        "persona_uuid": persona_uuid,
        "question_type": item.get("question_type", "unknown"),
        "answerability": answerability,
        "question": question,
        "ground_truth_answer": ground_truth,
        "answer_aliases": aliases,
        "predicted_answer": predicted,
        "api_status": api_status,
        "evaluation": evaluation,
        "trace": trace,
        "memory_read_result": {
            "backend": "memoryos",
            "stored_representation": "text_hierarchical",
            "retrieved_units": retrieved_units,
            "delivered_units": packed["delivered"],
            "backend_native_answer": predicted,
            "observability": {
                "write_provenance": False,
                "retrieval_ranks": True,
                "delivery_manifest": True,
            },
        },
        "latency_seconds": round(latency, 2),
    }


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate QA on pre-built MemoryOS memories"
    )
    p.add_argument("--config", required=True, help="Path to evaluation config JSON")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory for results")
    p.add_argument("--max-personas", type=int, default=None,
                   help="Limit personas to evaluate after filtering")
    p.add_argument("--eval-workers", type=int, default=None,
                   help="Override eval_workers from config")
    p.add_argument("--persona-filter", default=None,
                   help="Comma-separated persona UUID substrings (for testing subsets)")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    api_key = config["_resolved_api_key"]

    storage_path = config["memoryos"]["data_storage_path"]
    data_dir = config["data_dir"]
    canonical_records = load_canonical_records(config.get("canonical_data_dir"))
    canonical_by_persona = records_by_persona(canonical_records)
    selected_personas = [
        puid
        for puid in canonical_by_persona
        if not args.persona_filter
        or any(f.strip() in puid for f in args.persona_filter.split(","))
    ]
    if args.max_personas is not None:
        selected_personas = selected_personas[:args.max_personas]
    canonical_by_persona = {
        puid: canonical_by_persona[puid] for puid in selected_personas
    }
    if not canonical_by_persona:
        print("No canonical personas match the requested filters.")
        return

    # --- Discover built personas from manifest or disk ---
    manifest_path = os.path.join(storage_path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"ERROR: No manifest found at {manifest_path}")
        print("Run build_memories.py first to construct memories.")
        sys.exit(1)

    manifest = load_json(manifest_path)
    built_uuids = {p["persona_uuid"]: p for p in manifest["personas"]}
    print(f"Found {len(built_uuids)} built persona(s) from manifest")

    # --- Collect personas to evaluate ---
    to_eval: list[tuple[str, list[dict]]] = []
    missing_stores: list[tuple[str, list[dict]]] = []
    for puid, records in canonical_by_persona.items():
        if puid in built_uuids:
            to_eval.append((puid, records))
        else:
            missing_stores.append((puid, records))

    # --- Load all MemoryOS instances (auto-restore from disk) ---
    print(f"\nLoading {len(to_eval)} memory instance(s)...")

    # Preload embedding model once to avoid race in parallel workers.
    # If every selected store is missing, terminal records can be emitted
    # without loading a model at all.
    mcfg = config["memoryos"]
    if to_eval:
        from utils import get_embedding as _get_emb
        model_name = mcfg["embedding_model_name"]
        kwargs = mcfg.get("embedding_model_kwargs", {}) or {}
        print(f"  Preloading embedding model: {model_name} ...")
        _get_emb("warmup", model_name=model_name, **kwargs)
        print("  Embedding model ready.")
    else:
        print("  No prebuilt stores matched; emitting terminal records.")

    memory_pool: dict[str, MemoryOSEvalWrapper] = {}
    persona_meta: dict[str, dict] = {}
    load_failed: list[dict] = []

    def _load_one(puid: str, records: list[dict]) -> dict:
        wrapper = MemoryOSEvalWrapper(puid, api_key, config)
        return {
            "puid": puid,
            "wrapper": wrapper,
            "persona_name": records[0].get("persona_name", "unknown"),
            "file": records[0].get("source_filename", ""),
            "items": [record["item"] for record in records],
        }

    load_workers = max(1, min(len(to_eval), 8))
    with ThreadPoolExecutor(max_workers=load_workers) as load_executor:
        load_futures = {}
        for puid, records in to_eval:
            future = load_executor.submit(_load_one, puid, records)
            load_futures[future] = (puid, records)

        for future in as_completed(load_futures):
            puid, records = load_futures[future]
            try:
                result = future.result()
                memory_pool[result["puid"]] = result.pop("wrapper")
                persona_meta[result["puid"]] = result
            except Exception as e:
                load_failed.append({"persona_uuid": puid, "error": str(e)})

    terminal_results: list[dict] = []
    for puid, records in missing_stores:
        message = "Canonical persona has no prebuilt MemoryOS store"
        load_failed.append({"persona_uuid": puid, "error": message})
        terminal_results.extend(
            _terminal_error(record["item"], puid, message) for record in records
        )
    loaded_uuids = set(memory_pool)
    for failure in load_failed:
        puid = failure["persona_uuid"]
        if puid in loaded_uuids or puid not in canonical_by_persona:
            continue
        if any(result["persona_uuid"] == puid for result in terminal_results):
            continue
        terminal_results.extend(
            _terminal_error(record["item"], puid, failure["error"])
            for record in canonical_by_persona[puid]
        )

    print(f"  Loaded: {len(memory_pool)}, Failed: {len(load_failed)}")

    # --- Flatten all questions ---
    all_work: list[tuple[dict, str]] = []
    for puid, meta in persona_meta.items():
        for item in meta["items"]:
            all_work.append((item, puid))

    eval_workers = args.eval_workers or config.get("eval_workers", 100)
    rcfg = config["retrieval"]
    llm_cfg = config.get("llm", {})

    print(f"\n{'=' * 60}")
    print(f"Evaluating {len(all_work)} questions")
    print(f"  Workers: {eval_workers}")
    print(f"  Eval model: {llm_cfg.get('eval_model', config['memoryos']['llm_model'])}")
    print(f"{'=' * 60}")

    persona_item_results: dict[str, list[dict]] = defaultdict(list)
    for result in terminal_results:
        persona_item_results[result["persona_uuid"]].append(result)

    eval_start = time.time()
    eval_completed = 0
    eval_correct = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=len(all_work), unit="q", desc="Eval",
                     dynamic_ncols=True)
        _use_tqdm = True
    except ImportError:
        pbar = None
        _use_tqdm = False

    with ThreadPoolExecutor(max_workers=eval_workers) as executor:
        future_map = {}
        for item, puid in all_work:
            wrapper = memory_pool[puid]
            future = executor.submit(_eval_one, item, puid, wrapper, llm_cfg, rcfg)
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
            except Exception as e:
                item_result = _terminal_error(item, puid, e)
                persona_item_results[puid].append(item_result)
                if pbar:
                    pbar.write(f"ERROR: {item.get('question_id', '?')} -- {e}")
                else:
                    print(f"ERROR: {item.get('question_id', '?')} -- {e}")

            if pbar:
                elapsed = time.time() - eval_start
                rate = eval_completed / elapsed if elapsed > 0 else 0
                pbar.set_postfix(acc=f"{eval_correct / eval_completed:.3f}" if eval_completed else "?",
                                 qps=f"{rate:.0f}")
                pbar.update(1)
            elif eval_completed % 100 == 0 or eval_completed == len(all_work):
                elapsed = time.time() - eval_start
                rate = eval_completed / elapsed if elapsed > 0 else 0
                print(f"[{eval_completed}/{len(all_work)}] "
                      f"acc={eval_correct / eval_completed:.3f} "
                      f"({rate:.0f} q/s, {elapsed:.0f}s)")

    if pbar:
        pbar.close()

    eval_elapsed = time.time() - eval_start
    print(f"\nDone: {eval_completed} questions in {eval_elapsed:.0f}s "
          f"({eval_completed / eval_elapsed:.1f} q/s)")

    # --- Assemble results ---
    all_persona_results = []
    for puid, meta in persona_meta.items():
        items = persona_item_results.get(puid, [])
        built_info = built_uuids.get(puid, {})
        all_persona_results.append({
            "persona_uuid": puid,
            "persona_name": meta["persona_name"],
            "file": meta["file"],
            "num_qa_pairs_ingested": built_info.get("num_pairs_ingested", 0),
            "ingest_time_seconds": built_info.get("ingest_time_seconds", 0),
            "metrics": compute_metrics(items),
            "items": items,
        })

    # --- Write results ---
    output_dir = config.get("output_dir", "evaluation/memoryos/results/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "memoryos_mosaic_results.json")

    all_items = _collect_all_items(all_persona_results)
    # Include personas whose store failed to load in the same nested output.
    known_personas = {result["persona_uuid"] for result in all_persona_results}
    for puid, records in canonical_by_persona.items():
        if puid in known_personas or not persona_item_results.get(puid):
            continue
        all_persona_results.append({
            "persona_uuid": puid,
            "persona_name": records[0].get("persona_name", "unknown"),
            "file": records[0].get("source_filename", ""),
            "num_qa_pairs_ingested": 0,
            "ingest_time_seconds": 0,
            "metrics": compute_metrics(persona_item_results[puid]),
            "items": persona_item_results[puid],
        })
    all_items = _collect_all_items(all_persona_results)
    expected_qids = [
        record["question_id"] for record in canonical_records
        if record["persona_uuid"] in canonical_by_persona
    ]
    qid_order = {qid: index for index, qid in enumerate(expected_qids)}
    all_items.sort(key=lambda item: qid_order.get(item.get("question_id", ""), len(qid_order)))
    validate_result_qids(all_items, expected_qids)
    report = {
        "config_summary": {
            "data_dir": data_dir,
            "storage_path": storage_path,
            "memoryos_model": config["memoryos"]["llm_model"],
            "eval_model": llm_cfg.get("eval_model", config["memoryos"]["llm_model"]),
            "eval_workers": eval_workers,
        },
        "total_personas_evaluated": len(all_persona_results),
        "total_questions": len(all_items),
        "global_metrics": compute_metrics(all_items),
        "personas": all_persona_results,
        "load_failures": load_failed,
        "generated_at": datetime.now().isoformat(),
    }
    save_json(output_path, report)
    write_details_jsonl(output_dir, all_items)
    representation = "omni_text" if "omni" in os.path.basename(args.config).lower() else "caption_text"
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get("canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"),
        num_requested_items=len(expected_qids),
        evaluated_item_count=len(all_items),
        execution_track="native_end_to_end",
        memory_organization="text_hierarchical",
        representation=representation,
        index_manifest=manifest_path,
        retriever="MemoryOS retrieve_context",
        native_system_reader=llm_cfg.get("eval_model", config["memoryos"]["llm_model"]),
        chunk_unit="memory_page",
        query_policy="original",
        answer_reasoner=llm_cfg.get("eval_model", config["memoryos"]["llm_model"]),
        judge=config.get("judge"),
        top_k=rcfg.get("top_k_sessions"),
        delivery_budget={"mode": "production_budget", "media": "text_only"},
    )
    print(f"\nResults saved to: {output_path}")

    _print_summary(all_persona_results, load_failed, label="EVALUATION")


if __name__ == "__main__":
    main()
