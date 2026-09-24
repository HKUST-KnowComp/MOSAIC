#!/usr/bin/env python3
"""Replay memory-system retrieval exports through one fixed final reader."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from loguru import logger
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EVALUATION_ROOT = HERE.parent
for import_root in (str(EVALUATION_ROOT), str(HERE)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _common import load_config, match_answer  # noqa: E402
from answer_harness import ANSWER_PROMPT_VERSION, CommonAnswerHarness  # noqa: E402
from protocol import (  # noqa: E402
    DeliveryPolicy,
    canonical_qid_sha256,
    load_canonical_records,
    load_qid_manifest,
    make_item_trace,
    materialize_openai_media_blocks,
    pack_evidence,
    sha256_file,
    validate_result_qids,
    write_details_jsonl,
    write_run_manifest,
)
from run_stage_decomposition import (  # noqa: E402
    _fact_keys,
    _read_json,
    _require_api_key,
    _summary,
    _write_json_atomic,
)


def _read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _source_units(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize flat-RAG, A-MEM, and MemoryOS retrieval exports."""
    memory_read = row.get("memory_read_result") or {}
    # Prefer full-content exports. Flat-RAG `retrieved` truncates content to
    # 300 chars for logging; `trace.retrieval.ranked` / delivery keep full text.
    candidates = (
        memory_read.get("retrieved_units"),
        (row.get("trace") or {}).get("delivery", {}).get("delivered"),
        (row.get("trace") or {}).get("retrieval", {}).get("ranked"),
        row.get("retrieved"),
    )
    raw_units = next(
        (value for value in candidates if isinstance(value, list) and value),
        [],
    )
    units: list[dict[str, Any]] = []
    for rank, raw in enumerate(raw_units, start=1):
        if not isinstance(raw, Mapping):
            raw = {"content": str(raw)}
        units.append({
            "rank": rank,
            "chunk_id": str(raw.get("chunk_id") or raw.get("id") or f"unit:{rank}"),
            "session_id": str(
                raw.get("session_id") or raw.get("source_session_id") or ""
            ),
            "timestamp": raw.get("timestamp", raw.get("session_date", "")),
            "modality": str(raw.get("modality") or "text"),
            "content": str(raw.get("content") or raw.get("text") or ""),
            "fact_key": raw.get("fact_key"),
            "fact_keys": _fact_keys(raw),
            "score": raw.get("score", raw.get("similarity")),
            "media_path": raw.get("media_path"),
            **({"memory_tier": raw["memory_tier"]} if raw.get("memory_tier") else {}),
        })
    return units


def _source_observability(
    row: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, bool]:
    exported = (
        (row.get("memory_read_result") or {}).get("observability") or {}
    )
    configured = source.get("observability") or {}
    return {
        "write_provenance": bool(exported.get(
            "write_provenance", configured.get("write_provenance", True)
        )),
        "retrieval_ranks": bool(exported.get(
            "retrieval_ranks", configured.get("retrieval_ranks", True)
        )),
        "delivery_manifest": True,
    }


def _validate_source_manifest(
    details_path: Path,
    canonical_manifest_path: str,
    *,
    allow_legacy: bool,
) -> dict[str, Any]:
    source_manifest_path = details_path.parent / "run_manifest.json"
    canonical = load_qid_manifest(canonical_manifest_path)
    if not source_manifest_path.is_file():
        if allow_legacy:
            return {
                "status": "legacy_unverified",
                "path": None,
                "sha256": None,
            }
        raise FileNotFoundError(
            f"Source run manifest is required: {source_manifest_path}. "
            "Rerun the source export or pass --allow-legacy-source explicitly."
        )
    source_manifest = _read_json(source_manifest_path)
    source_hash = source_manifest.get("canonical_qid_sha256")
    if source_hash != canonical["question_id_sha256"]:
        raise ValueError(
            f"Source/canonical QID manifest mismatch: {source_manifest_path}"
        )
    return {
        "status": "verified",
        "path": str(source_manifest_path),
        "sha256": sha256_file(source_manifest_path),
        "manifest": source_manifest,
    }


def load_source_export(
    source: Mapping[str, Any],
    canonical_manifest_path: str,
    *,
    allow_legacy: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    details_path = Path(str(source["details_path"]))
    if not details_path.is_file():
        raise FileNotFoundError(
            f"Retrieval export not found: {details_path}. Run its documented "
            "source-export command before fixed-reader replay."
        )
    provenance = _validate_source_manifest(
        details_path, canonical_manifest_path, allow_legacy=allow_legacy
    )
    canonical = load_qid_manifest(canonical_manifest_path)
    rows = _read_jsonl(details_path)
    validate_result_qids(rows, list(canonical["question_ids"]))
    by_qid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        qid = str(row.get("question_id") or row.get("item_id") or "")
        if not qid:
            raise ValueError(f"Source export contains a row without QID: {details_path}")
        if (
            source.get("memory_system") in {"a_mem", "memoryos"}
            and not row.get("memory_read_result")
            and not allow_legacy
        ):
            raise ValueError(
                f"Source export lacks memory_read_result observability: "
                f"{details_path} ({qid})"
            )
        if qid in by_qid:
            duplicates.append(qid)
        by_qid[qid] = row
    if duplicates:
        raise ValueError(f"Source export contains duplicate QIDs: {duplicates[:5]}")
    source_manifest = provenance.get("manifest") or {}
    if source_manifest and int(
        source_manifest.get("evaluated_item_count", len(rows))
    ) != len(rows):
        raise ValueError(
            f"Source run manifest/result count mismatch: {details_path}"
        )
    provenance.update({
        "details_path": str(details_path),
        "details_sha256": sha256_file(details_path),
        "num_source_rows": len(rows),
    })
    return by_qid, provenance


def _select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    qid_manifest: str | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    selected = [
        dict(record) for record in records
        if record.get("answerability") == "answerable"
    ]
    if qid_manifest:
        payload = _read_json(qid_manifest)
        requested = payload.get("question_ids", payload)
        if not isinstance(requested, list):
            raise ValueError("QID manifest must be a list or contain question_ids")
        by_qid = {str(record["question_id"]): record for record in selected}
        missing = [str(qid) for qid in requested if str(qid) not in by_qid]
        if missing:
            raise ValueError(f"Pilot QID manifest contains unavailable IDs: {missing[:5]}")
        selected = [dict(by_qid[str(qid)]) for qid in requested]
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def _delivery_policy(config: Mapping[str, Any]) -> DeliveryPolicy:
    budget = dict(config.get("delivery") or {})
    budget["mode"] = "production_budget"
    return DeliveryPolicy.from_config(budget)


class FixedReaderRunner:
    def __init__(
        self,
        config: Mapping[str, Any],
        condition: str,
        source: Mapping[str, Any],
        source_rows: Mapping[str, Mapping[str, Any]],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        self.config = dict(config)
        self.condition = condition
        self.source = dict(source)
        self.source_rows = dict(source_rows)
        self.records = [dict(record) for record in records]
        self.harness = CommonAnswerHarness(self.config["reader"])
        self.policy = _delivery_policy(self.config)
        self.top_k = int(self.config.get("top_k", 20))

    def build_context(
        self, record: Mapping[str, Any], *, materialize: bool
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
        qid = str(record["question_id"])
        source_row = self.source_rows.get(qid)
        if source_row is None:
            raise KeyError(f"Source retrieval export is missing {qid}")
        units = _source_units(source_row)[: self.top_k]
        if not units and source_row.get("api_status") == "error":
            raise ValueError(
                f"Source retrieval failed for {qid}: {source_row.get('error', 'unknown error')}"
            )
        delivery = pack_evidence(units, self.policy)
        if materialize and any(
            unit.get("native_attached") for unit in delivery.get("delivered", [])
        ):
            delivery = materialize_openai_media_blocks(
                delivery, api_format=self.config["reader"].get("api_format")
            )
        else:
            delivery["provider_blocks"] = []
        return units, delivery, _source_observability(source_row, self.source)

    def terminal(self, record: Mapping[str, Any], error: Exception | str) -> dict[str, Any]:
        item = record["item"]
        message = str(error)
        return {
            "question_id": record["question_id"],
            "persona_uuid": record["persona_uuid"],
            "question_type": item.get("question_type"),
            "answerability": "answerable",
            "question": item.get("question"),
            "expected_answer": item.get("answer"),
            "answer_aliases": item.get("answer_aliases", []),
            "condition": self.condition,
            "memory_system": self.source.get("memory_system"),
            "representation": self.source.get("representation"),
            "predicted_answer": "",
            "api_status": "error",
            "evaluation": {
                "exact_match": False,
                "alias_match": False,
                "unanswerable_correct": False,
            },
            "error": message,
            "trace": make_item_trace(
                question_id=str(record["question_id"]),
                query={"original": item.get("question", ""),
                       "retrieval_query": item.get("question", "")},
                evidence_plan=item.get("evidence_plan", []),
                answer={
                    "predicted": "", "api_status": "error", "forced": False,
                    "error_class": type(error).__name__ if isinstance(error, Exception) else "Error",
                    "error_message": message,
                },
                required_session_ids=item.get("answer_session_ids", []),
            ),
        }

    def evaluate_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time()
        item = record["item"]
        try:
            units, delivery, observability = self.build_context(
                record, materialize=True
            )
            answer = self.harness.answer(
                str(item.get("question") or ""),
                delivery.get("delivered", []),
                delivery.get("provider_blocks", []),
                question_type=item.get("question_type"),
            )
            evaluation = (
                match_answer(
                    answer.text, str(item.get("answer") or ""),
                    item.get("answer_aliases", []), "answerable",
                )
                if answer.api_status == "ok"
                else {"exact_match": False, "alias_match": False,
                      "unanswerable_correct": False}
            )
            source_row = self.source_rows[str(record["question_id"])]
            source_trace = source_row.get("trace") or {}
            source_query = source_trace.get("query") or {}
            trace = make_item_trace(
                question_id=str(record["question_id"]),
                query={
                    "original": item.get("question", ""),
                    "retrieval_query": source_query.get(
                        "retrieval_query", item.get("question", "")
                    ),
                    "rewrite_model": source_query.get("rewrite_model"),
                },
                ranked=units,
                evidence_plan=item.get("evidence_plan", []),
                delivery=delivery,
                answer={
                    "predicted": answer.text,
                    "api_status": answer.api_status,
                    "forced": False,
                    "attempts": answer.attempts,
                    "finish_reason": answer.finish_reason,
                    "error_class": answer.error_class,
                    "error_message": answer.error_message,
                },
                required_session_ids=item.get("answer_session_ids", []),
            )
            if not observability["write_provenance"]:
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
                "question_id": record["question_id"],
                "persona_uuid": record["persona_uuid"],
                "question_type": item.get("question_type"),
                "answerability": "answerable",
                "question": item.get("question"),
                "expected_answer": item.get("answer"),
                "answer_aliases": item.get("answer_aliases", []),
                "condition": self.condition,
                "memory_system": self.source.get("memory_system"),
                "representation": self.source.get("representation"),
                "predicted_answer": answer.text,
                "api_status": answer.api_status,
                "evaluation": evaluation,
                "trace": trace,
                "source_observability": observability,
                "source_native_answer": (
                    (source_row.get("memory_read_result") or {}).get(
                        "backend_native_answer"
                    ) or source_row.get("predicted_answer")
                ),
                "usage": {
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "total_tokens": answer.total_tokens,
                    "estimated_cost_usd": answer.estimated_cost_usd,
                },
                "latency_seconds": round(time.time() - started, 3),
            }
        except Exception as error:
            logger.warning("{} failed: {}", record["question_id"], error)
            return self.terminal(record, error)

    def run(self, workers: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if workers <= 1:
            for record in tqdm(self.records, desc=self.condition, unit="item"):
                results.append(self.evaluate_one(record))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.evaluate_one, record): record
                    for record in self.records
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=self.condition, unit="item",
                ):
                    record = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(self.terminal(record, error))
        order = {
            str(record["question_id"]): index
            for index, record in enumerate(self.records)
        }
        results.sort(key=lambda row: order[str(row["question_id"])])
        validate_result_qids(results, list(order))
        return results


def _load_resume(path: Path, records: Sequence[Mapping[str, Any]],
                 condition: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows = _read_jsonl(path)
    expected = [str(record["question_id"]) for record in records]
    validate_result_qids(rows, expected)
    if any(row.get("condition") != condition for row in rows):
        raise ValueError("Resume file contains a different fixed-reader condition")
    failed = {
        str(row["question_id"]) for row in rows
        if row.get("api_status") in {"error", "timeout", "empty"} or row.get("error")
    }
    return rows, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EVALUATION_ROOT / "configs" / "fixed_reader_comparison.json"),
    )
    parser.add_argument("--condition", required=True)
    parser.add_argument("--qid-manifest", help="Ordered pilot question-ID JSON")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--allow-legacy-source", action="store_true",
        help="Allow an old retrieval export without run_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sources = config.get("sources") or {}
    if args.condition not in sources:
        raise ValueError(
            f"Unknown condition {args.condition!r}; choices={sorted(sources)}"
        )
    source = dict(sources[args.condition])
    canonical = load_canonical_records(config.get("canonical_data_dir"))
    records = _select_records(
        canonical, qid_manifest=args.qid_manifest, max_items=args.max_items
    )
    if not records:
        raise SystemExit("No answerable records selected")
    source_rows, source_provenance = load_source_export(
        source,
        config.get("canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"),
        allow_legacy=args.allow_legacy_source,
    )
    missing = [
        str(record["question_id"]) for record in records
        if str(record["question_id"]) not in source_rows
    ]
    if missing:
        raise ValueError(f"Retrieval export is missing selected QIDs: {missing[:5]}")

    output_dir = Path(
        args.output_dir or Path(config["output_root"]) / args.condition
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "run.log", level="DEBUG")
    runner = FixedReaderRunner(
        config, args.condition, source, source_rows, records
    )

    if args.validate_only:
        contexts = []
        for record in tqdm(records, desc="Validate replay", unit="item"):
            units, delivery, observability = runner.build_context(
                record, materialize=False
            )
            contexts.append({
                "question_id": record["question_id"],
                "num_source_units": len(_source_units(
                    source_rows[str(record["question_id"])]
                )),
                "num_budgeted_units": len(units),
                "num_delivered": len(delivery.get("delivered", [])),
                "num_dropped": len(delivery.get("dropped", [])),
                "drop_reasons": delivery.get("drop_reasons", {}),
                "observability": observability,
            })
        _write_json_atomic(output_dir / "context_validation.json", {
            "condition": args.condition,
            "num_items": len(contexts),
            "question_id_sha256": canonical_qid_sha256(
                row["question_id"] for row in contexts
            ),
            "source_provenance": source_provenance,
            "items": contexts,
        })
        print(f"Validated {len(contexts)} fixed-reader contexts: {output_dir}")
        return

    _require_api_key(config["reader"], "Reader")
    existing: list[dict[str, Any]] | None = None
    records_to_run = records
    details_path = output_dir / "details.jsonl"
    if args.resume:
        if not details_path.is_file():
            raise FileNotFoundError(f"Resume requested but missing {details_path}")
        existing, failed = _load_resume(details_path, records, args.condition)
        if not failed:
            print(f"No terminal failures to resume: {details_path}")
            return
        records_to_run = [
            record for record in records
            if str(record["question_id"]) in failed
        ]
        runner.records = records_to_run

    results = runner.run(int(args.workers or config.get("workers", 10)))
    if existing is not None:
        replacements = {str(row["question_id"]): row for row in results}
        results = [
            replacements.get(str(row["question_id"]), row) for row in existing
        ]
        validate_result_qids(
            results, [str(record["question_id"]) for record in records]
        )
    write_details_jsonl(output_dir, results)
    summary = _summary(results)
    summary.update({
        "condition": args.condition,
        "memory_system": source.get("memory_system"),
        "representation": source.get("representation"),
        "source_provenance": source_provenance,
    })
    _write_json_atomic(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get(
            "canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"
        ),
        num_requested_items=len(records),
        evaluated_item_count=len(results),
        execution_track="fixed_reader_architecture_comparison",
        memory_organization=source.get("memory_organization"),
        representation=source.get("representation"),
        index_manifest=source_provenance.get("path"),
        retriever=source.get("retriever"),
        native_system_reader=None,
        chunk_unit=source.get("chunk_unit"),
        query_policy="source_system",
        answer_reasoner=config["reader"].get("llm_model"),
        judge=config.get("judge"),
        top_k=config.get("top_k", 20),
        delivery_budget=dict(config.get("delivery") or {}),
        controlled_qid_sha256=canonical_qid_sha256(
            record["question_id"] for record in records
        ),
        answer_prompt_version=ANSWER_PROMPT_VERSION,
        source_details_path=source_provenance["details_path"],
        source_details_sha256=source_provenance["details_sha256"],
        source_run_manifest_sha256=source_provenance.get("sha256"),
        source_observability=source.get("observability", {}),
    )
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
