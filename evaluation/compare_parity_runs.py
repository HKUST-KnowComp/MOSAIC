#!/usr/bin/env python3
"""Compare a Naive RAG run with ReAct one-search parity output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _load_details(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _chunk_id(unit: Mapping[str, Any]) -> str:
    return str(unit.get("chunk_id") or unit.get("id") or "")


def _delivery_signature(trace: Mapping[str, Any]) -> dict[str, Any]:
    delivery = trace.get("delivery") or {}
    return {
        "delivered": [
            (
                _chunk_id(unit),
                unit.get("delivery_status"),
                bool(unit.get("native_attached")),
            )
            for unit in delivery.get("delivered", [])
        ],
        "dropped": [
            (_chunk_id(unit), unit.get("drop_reason"))
            for unit in delivery.get("dropped", [])
        ],
        "failed": [
            (_chunk_id(unit), unit.get("drop_reason"))
            for unit in delivery.get("failed", [])
        ],
        "drop_reasons": delivery.get("drop_reasons", {}),
        "attachment_counts": delivery.get("attachment_counts", {}),
        "effective_context_chars": delivery.get("effective_context_chars", 0),
    }


def compare_records(
    baseline: Sequence[Mapping[str, Any]],
    parity: Sequence[Mapping[str, Any]],
    *,
    compare_predictions: bool = True,
) -> dict[str, Any]:
    baseline_qids = [str(row.get("question_id") or "") for row in baseline]
    parity_qids = [str(row.get("question_id") or "") for row in parity]
    mismatches: list[dict[str, Any]] = []
    if baseline_qids != parity_qids:
        mismatches.append({
            "field": "question_id_order",
            "baseline_count": len(baseline_qids),
            "parity_count": len(parity_qids),
        })
        return {
            "passed": False,
            "num_items": min(len(baseline), len(parity)),
            "num_mismatches": len(mismatches),
            "mismatches": mismatches,
        }

    for baseline_row, parity_row in zip(baseline, parity):
        question_id = str(baseline_row.get("question_id") or "")
        baseline_trace = baseline_row.get("trace") or {}
        parity_trace = parity_row.get("trace") or {}
        comparisons = {
            "query": (
                baseline_trace.get("query"),
                parity_trace.get("query"),
            ),
            "retrieved_ids": (
                [_chunk_id(unit) for unit in (baseline_trace.get("retrieval") or {}).get("ranked", [])],
                [_chunk_id(unit) for unit in (parity_trace.get("retrieval") or {}).get("ranked", [])],
            ),
            "delivery": (
                _delivery_signature(baseline_trace),
                _delivery_signature(parity_trace),
            ),
            "api_status": (
                baseline_row.get("api_status"),
                parity_row.get("api_status"),
            ),
        }
        if compare_predictions:
            comparisons["predicted_answer"] = (
                baseline_row.get("predicted_answer"),
                parity_row.get("predicted_answer"),
            )
        for field, (baseline_value, parity_value) in comparisons.items():
            if baseline_value != parity_value:
                mismatches.append({
                    "question_id": question_id,
                    "field": field,
                    "baseline": baseline_value,
                    "parity": parity_value,
                })

    return {
        "passed": not mismatches,
        "num_items": len(baseline),
        "num_mismatches": len(mismatches),
        "mismatches": mismatches[:50],
        "mismatches_truncated": max(0, len(mismatches) - 50),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("parity_dir", type=Path)
    parser.add_argument(
        "--ignore-predictions",
        action="store_true",
        help="Check query/retrieval/delivery plumbing without prediction equality",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    baseline_manifest = _load_json(args.baseline_dir / "run_manifest.json")
    parity_manifest = _load_json(args.parity_dir / "run_manifest.json")
    manifest_fields = ("representation", "index_manifest", "query_policy", "answer_reasoner", "top_k")
    manifest_mismatches = [
        field for field in manifest_fields
        if baseline_manifest.get(field) != parity_manifest.get(field)
    ]
    result = compare_records(
        _load_details(args.baseline_dir / "details.jsonl"),
        _load_details(args.parity_dir / "details.jsonl"),
        compare_predictions=not args.ignore_predictions,
    )
    result["manifest_fields_checked"] = list(manifest_fields)
    result["manifest_mismatches"] = manifest_mismatches
    result["passed"] = bool(result["passed"] and not manifest_mismatches)

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
