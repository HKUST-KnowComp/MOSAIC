#!/usr/bin/env python3
"""Build the retained MOSAIC paper tables from judged experiment outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
EVALUATION_ROOT = HERE.parent
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from _common import load_config  # noqa: E402
from protocol import (  # noqa: E402
    canonical_qid_sha256,
    load_canonical_records,
    load_qid_manifest,
    sha256_file,
    sha256_json,
)


GROUP_ORDER = (
    "end_to_end",
    "stage_decomposition",
    "stage_budget_sweep",
    "stage_second_reader_mimo",
    "stage_second_reader_gemini",
    "modality_interventions",
    "modality_second_reader_mimo",
    "modality_second_reader_gemini",
    "fixed_reader",
)

CONTROLLED_GROUPS = {
    "stage_decomposition", "stage_budget_sweep",
    "stage_second_reader_mimo", "stage_second_reader_gemini",
    "modality_interventions", "modality_second_reader_mimo",
    "modality_second_reader_gemini", "fixed_reader",
}

# P4 reader-replication groups. Skippable when the MiMo second-reader /
# Gemini pilot pair is unavailable (OpenRouter outages, deferred pilots).
SECOND_READER_GROUPS = (
    "stage_second_reader_mimo",
    "stage_second_reader_gemini",
    "modality_second_reader_mimo",
    "modality_second_reader_gemini",
)

STAGE_RETRIEVED_RUNS = {
    "retrieved_raw_all_delivered", "retrieved_raw_budgeted", "end_to_end",
}

REQUIRED_LEAK_AUDIT_VERSION = 2

REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "cannot determine",
    "cannot answer",
    "not enough information",
    "insufficient information",
    "unable to determine",
    "unable to answer",
    "unanswerable",
)

STAGE_COMPARISONS = (
    ("perception_cost", "gold_symbolic_facts", "gold_raw_evidence"),
    ("avoidable_write_loss", "gold_structured_text", "gold_generic_text"),
    ("retrieval_loss", "gold_raw_evidence", "retrieved_raw_all_delivered"),
    ("delivery_loss", "retrieved_raw_all_delivered", "retrieved_raw_budgeted"),
    ("production_query_effect", "retrieved_raw_budgeted", "end_to_end"),
    ("full_pipeline_loss", "gold_symbolic_facts", "end_to_end"),
)

INTERVENTION_COMPARISONS = (
    ("image_raw_minus_ocr", "image_raw", "image_ocr_only"),
    ("image_raw_minus_object", "image_raw", "image_object_only"),
    ("image_raw_minus_spatial", "image_raw", "image_spatial_description"),
    (
        "image_spatial_minus_object",
        "image_spatial_description",
        "image_object_only",
    ),
    ("video_raw_minus_sparse", "video_raw", "video_sparse_ordered_frames"),
    (
        "video_order_benefit",
        "video_sparse_ordered_frames",
        "video_shuffled_frames",
    ),
    (
        "video_raw_minus_event_description",
        "video_raw",
        "video_ordered_event_count_description",
    ),
    (
        "video_event_description_minus_sparse",
        "video_ordered_event_count_description",
        "video_sparse_ordered_frames",
    ),
    ("audio_raw_minus_asr", "audio_raw", "audio_asr"),
    ("audio_prosody_tag_recovery", "audio_asr_prosody", "audio_asr"),
    ("audio_raw_minus_prosody_tag", "audio_raw", "audio_asr_prosody"),
)

FIXED_READER_COMPARISONS = (
    ("amem_minus_flat_caption", "amem_caption_text", "flat_rag_caption_text"),
    (
        "memoryos_minus_flat_caption",
        "memoryos_caption_text",
        "flat_rag_caption_text",
    ),
    ("amem_minus_flat_omni", "amem_omni_text", "flat_rag_omni_text"),
    (
        "memoryos_minus_flat_omni",
        "memoryos_omni_text",
        "flat_rag_omni_text",
    ),
)


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (
                    json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                )
                for key in keys
            })


def _score(row: Mapping[str, Any]) -> float:
    if row.get("judge_error"):
        raise ValueError(f"Judge error for {row.get('question_id', '?')}")
    value = row.get("judge_score")
    if value is None:
        raise ValueError(f"Missing judge_score for {row.get('question_id', '?')}")
    score = float(value)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score outside [0,1]: {score}")
    return score


def _is_refusal(value: Any) -> bool:
    normalized = " ".join(str(value or "").strip().lower().split())
    return any(pattern in normalized for pattern in REFUSAL_PATTERNS)


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_seed(seed: int, label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:16], 16)
    return seed ^ suffix


def _bootstrap_mean(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not values:
        return []
    rng = random.Random(seed)
    length = len(values)
    return [
        sum(values[rng.randrange(length)] for _ in range(length)) / length
        for _ in range(samples)
    ]


def _cluster_bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    if not values:
        return []
    if len(values) != len(clusters):
        raise ValueError("Bootstrap values and cluster labels differ in length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    labels = list(grouped)
    if not labels:
        return []
    rng = random.Random(seed)
    output: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        for _ in labels:
            sampled.extend(grouped[labels[rng.randrange(len(labels))]])
        output.append(sum(sampled) / len(sampled))
    return output


def _interval(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    return [round(_quantile(values, 0.025), 6), round(_quantile(values, 0.975), 6)]


def estimate_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    scores = [_score(row) for row in rows]
    clusters = [str(row.get("persona_uuid") or row.get("question_id") or "") for row in rows]
    item_draws = _bootstrap_mean(
        scores, samples=samples, seed=_bootstrap_seed(seed, label + ":item")
    )
    cluster_draws = _cluster_bootstrap_mean(
        scores, clusters, samples=samples,
        seed=_bootstrap_seed(seed, label + ":persona"),
    )
    return {
        "n": len(rows),
        "mean_judge_score": _mean(scores),
        "item_bootstrap_95_ci": _interval(item_draws),
        "persona_clustered_bootstrap_95_ci": _interval(cluster_draws),
        "num_personas": len(set(clusters)),
        "bootstrap_samples": samples,
    }


def _ordered_ids(rows: Sequence[Mapping[str, Any]], id_field: str) -> list[str]:
    return [str(row.get(id_field) or "") for row in rows]


def validate_pair(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
) -> list[str]:
    left_ids = _ordered_ids(left, id_field)
    right_ids = _ordered_ids(right, id_field)
    if not all(left_ids) or not all(right_ids):
        raise ValueError(f"Paired rows contain empty {id_field}")
    if len(left_ids) != len(set(left_ids)):
        raise ValueError(f"Left run contains duplicate {id_field} values")
    if len(right_ids) != len(set(right_ids)):
        raise ValueError(f"Right run contains duplicate {id_field} values")
    if left_ids != right_ids:
        missing_left = sorted(set(right_ids) - set(left_ids))
        missing_right = sorted(set(left_ids) - set(right_ids))
        raise ValueError(
            f"Paired {id_field} sets/order differ: left={len(left_ids)}, "
            f"right={len(right_ids)}, missing_left={missing_left[:3]}, "
            f"missing_right={missing_right[:3]}"
        )
    return left_ids


def paired_effect(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    ids = validate_pair(left, right, id_field=id_field)
    differences = [_score(a) - _score(b) for a, b in zip(left, right)]
    clusters = [
        str(a.get("persona_uuid") or a.get("question_id") or identifier)
        for a, identifier in zip(left, ids)
    ]
    item_draws = _bootstrap_mean(
        differences, samples=samples,
        seed=_bootstrap_seed(seed, label + ":item"),
    )
    cluster_draws = _cluster_bootstrap_mean(
        differences, clusters, samples=samples,
        seed=_bootstrap_seed(seed, label + ":persona"),
    )
    tolerance = 1e-12
    return {
        "n": len(ids),
        "left_mean": _mean([_score(row) for row in left]),
        "right_mean": _mean([_score(row) for row in right]),
        "mean_paired_effect_left_minus_right": _mean(differences),
        "item_bootstrap_95_ci": _interval(item_draws),
        "persona_clustered_bootstrap_95_ci": _interval(cluster_draws),
        "wins": sum(value > tolerance for value in differences),
        "ties": sum(abs(value) <= tolerance for value in differences),
        "losses": sum(value < -tolerance for value in differences),
        "num_personas": len(set(clusters)),
        "bootstrap_samples": samples,
    }


def _required_entries(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(entry) for entry in item.get("evidence_plan", [])
        if entry.get("required_for_answer") and entry.get("available_at_query", True)
    ]


def canonical_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for record in records:
        item = record["item"]
        entries = _required_entries(item)
        modalities = sorted({str(entry.get("modality") or "text") for entry in entries})
        challenges: set[str] = set()
        for entry in entries:
            modality = str(entry.get("modality") or "")
            if modality in {"image", "video"}:
                challenge = str(entry.get("visual_gap_strategy") or "")
                if challenge and challenge != "empty":
                    challenges.add(challenge)
            elif modality == "audio":
                tone = str(entry.get("audio_tone") or "")
                if tone in {"happy", "unhappy"}:
                    challenges.add("prosody")
        metadata[str(record["question_id"])] = {
            "persona_uuid": str(record["persona_uuid"]),
            "question_type": item.get("question_type", "unknown"),
            "answerability": item.get("answerability", "unknown"),
            "required_modalities": modalities,
            "challenges": sorted(challenges),
            "chain_length": len(entries),
        }
    return metadata


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    source: str = "judged rows",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        qid = str(row.get("question_id") or row.get("item_id") or "")
        info = metadata.get(qid)
        if info is None:
            raise ValueError(f"Judged row has non-canonical QID: {qid}")
        for field in ("persona_uuid", "question_type", "answerability"):
            observed = row.get(field)
            expected = info[field]
            if observed is not None and observed != "" and str(observed) != str(expected):
                raise ValueError(
                    f"Canonical {field} mismatch for {qid} in {source}: "
                    f"observed={observed!r}, expected={expected!r}"
                )
            row[field] = expected
        row["question_id"] = qid
        api_status = str(row.get("api_status") or "missing")
        if api_status != "ok":
            raise ValueError(
                f"Non-ok inference api_status for {qid} in {source}: "
                f"{api_status!r}"
            )
        row["required_modalities"] = list(info["required_modalities"])
        row["canonical_challenges"] = list(info["challenges"])
        row["chain_length"] = info["chain_length"]
        _score(row)
        output.append(row)
    return output


def _validate_judge_summary(
    summary: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping):
        raise ValueError(f"Invalid judge summary for {label}")
    if "num_items" not in summary or "errors" not in summary:
        raise ValueError(
            f"Judge summary lacks num_items/errors for {label}"
        )
    try:
        num_items = int(summary["num_items"])
        declared_errors = int(summary["errors"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid judge summary counts for {label}"
        ) from error
    actual_errors = sum(bool(row.get("judge_error")) for row in rows)
    if num_items != len(rows):
        raise ValueError(
            f"Judge summary/result count mismatch for {label}: "
            f"summary={num_items}, rows={len(rows)}"
        )
    if declared_errors != actual_errors:
        raise ValueError(
            f"Judge summary error-count mismatch for {label}: "
            f"summary={declared_errors}, rows={actual_errors}"
        )
    if actual_errors:
        raise ValueError(f"Judge errors are present for {label}: {actual_errors}")
    return summary


def _validate_intervention_audit(
    result_dir: Path,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    run_name: str,
) -> dict[str, Any]:
    audit_value = manifest.get("leak_audit")
    if not audit_value:
        raise ValueError(f"Missing leak-audit reference for {run_name}")
    audit_path = Path(str(audit_value))
    if not audit_path.is_absolute():
        audit_path = result_dir / audit_path
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing intervention leak audit for {run_name}: {audit_path}"
        )
    audit = _read_json(audit_path)
    if not isinstance(audit, Mapping):
        raise ValueError(f"Invalid intervention leak audit: {audit_path}")
    if audit.get("audit_version") != REQUIRED_LEAK_AUDIT_VERSION:
        raise ValueError(
            f"Unsupported leak-audit version for {run_name}: "
            f"{audit.get('audit_version')!r}"
        )
    if audit.get("passed") is not True:
        raise ValueError(f"Intervention leak audit failed for {run_name}")
    if audit.get("condition") != run_name:
        raise ValueError(
            f"Leak-audit condition mismatch for {run_name}: "
            f"{audit.get('condition')!r}"
        )
    try:
        audit_count = int(audit.get("num_cases", -1))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid leak-audit case count for {run_name}"
        ) from error
    if audit_count != len(rows):
        raise ValueError(
            f"Leak-audit/result count mismatch for {run_name}"
        )
    row_ids = _ordered_ids(rows, "intervention_id")
    audit_items = audit.get("items")
    if not isinstance(audit_items, list):
        raise ValueError(f"Leak audit lacks an item list for {run_name}")
    if any(
        not isinstance(item, Mapping)
        or item.get("audit_version") != REQUIRED_LEAK_AUDIT_VERSION
        or item.get("status") != "complete"
        or item.get("passed") is not True
        for item in audit_items
    ):
        raise ValueError(
            f"Leak audit contains incomplete or failed items for {run_name}"
        )
    audit_ids = [
        str(item.get("intervention_id") or "")
        for item in audit_items
        if isinstance(item, Mapping)
    ]
    if audit_ids != row_ids:
        raise ValueError(
            f"Leak-audit intervention IDs differ for {run_name}"
        )
    actual_sha256 = sha256_file(audit_path)
    if manifest.get("leak_audit_sha256") != actual_sha256:
        raise ValueError(
            f"Leak-audit hash mismatch for {run_name}"
        )
    return {
        "leak_audit": str(audit_path),
        "leak_audit_sha256": actual_sha256,
    }


def _result_paths(run_config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    result_dir = Path(str(run_config["result_dir"]))
    judged = Path(str(
        run_config.get("judged_details_path")
        or result_dir / str(run_config.get("judge_subdir", "judge")) / "details_judged.jsonl"
    ))
    judge_summary = Path(str(
        run_config.get("judge_summary_path")
        or judged.parent / "summary_judged.json"
    ))
    manifest = Path(str(
        run_config.get("run_manifest_path") or result_dir / "run_manifest.json"
    ))
    return judged, judge_summary, manifest


def _validate_judged_binding(
    inference_rows: Sequence[Mapping[str, Any]],
    judged_rows: Sequence[Mapping[str, Any]],
    *,
    id_field: str,
    label: str,
) -> None:
    inference_ids = _ordered_ids(inference_rows, id_field)
    judged_ids = _ordered_ids(judged_rows, id_field)
    if inference_ids != judged_ids:
        raise ValueError(f"Inference/judge ID order differs for {label}")
    for inference, judged in zip(inference_rows, judged_rows):
        for field in (
            "question", "predicted_answer", "api_status",
            "expected_answer", "ground_truth_answer",
        ):
            if field in inference and inference.get(field) != judged.get(field):
                raise ValueError(
                    f"Stale judged input for {label}: {field} differs at "
                    f"{inference.get(id_field)}"
                )


def _manifest_compatibility_value(manifest: Mapping[str, Any], key: str) -> Any:
    value = manifest.get(key)
    if key == "judge" and isinstance(value, Mapping):
        return value.get("llm_model") or value.get("model")
    return value


def _validate_expected_delivery(
    run_config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> None:
    expected = run_config.get("expected_delivery_budget")
    if not isinstance(expected, Mapping):
        return
    actual = manifest.get("delivery_budget") or {}
    policy = actual.get("policy") or {}
    observed = {
        "production_top_k": actual.get("production_top_k"),
        "max_images": policy.get("max_images"),
        "max_audio": policy.get("max_audio"),
        "max_video": policy.get("max_video"),
    }
    mismatches = {
        key: {"expected": value, "actual": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Delivery-budget manifest mismatch for {label}: {mismatches}"
        )


def _validate_stage_identity(
    group: str,
    run_name: str,
    manifest: Mapping[str, Any],
) -> None:
    if not group.startswith("stage_"):
        return
    expected_condition = (
        "retrieved_raw_budgeted"
        if group == "stage_budget_sweep" else run_name
    )
    delivery = manifest.get("delivery_budget") or {}
    expected_query_policy = (
        "fixed_rewrite" if expected_condition == "end_to_end" else "original"
    )
    observed = {
        "representation": manifest.get("representation"),
        "condition": delivery.get("condition"),
        "query_policy": manifest.get("query_policy"),
    }
    expected = {
        "representation": expected_condition,
        "condition": expected_condition,
        "query_policy": expected_query_policy,
    }
    if observed != expected:
        raise ValueError(
            f"Stage run identity mismatch for {group}/{run_name}: "
            f"expected={expected}, actual={observed}"
        )


def _compatible_fields(group: str) -> tuple[str, ...]:
    if group in {
        "stage_decomposition", "stage_second_reader_mimo",
        "stage_second_reader_gemini",
    }:
        return (
            "canonical_qid_sha256", "controlled_qid_sha256",
            "answer_reasoner", "answer_prompt_version", "judge",
            "config_sha256",
        )
    if group == "stage_budget_sweep":
        return (
            "canonical_qid_sha256", "controlled_qid_sha256",
            "answer_reasoner", "answer_prompt_version", "judge",
            "retrieval_manifest_sha256",
        )
    if group == "fixed_reader":
        return (
            "canonical_qid_sha256", "controlled_qid_sha256",
            "answer_reasoner", "answer_prompt_version", "judge", "top_k",
            "delivery_budget", "config_sha256",
        )
    if group in {
        "modality_interventions", "modality_second_reader_mimo",
        "modality_second_reader_gemini",
    }:
        return (
            "canonical_qid_sha256", "controlled_intervention_id_sha256",
            "answer_reasoner", "answer_prompt_version", "judge",
            "transformation_version", "challenges", "config_sha256",
        )
    return ("canonical_qid_sha256", "judge")


def _resolve_skip_groups(
    config: Mapping[str, Any],
    *,
    skip_second_reader: bool,
    skip_groups: Sequence[str],
) -> list[str]:
    skipped: list[str] = []
    for value in config.get("skip_groups") or []:
        skipped.append(str(value))
    if skip_second_reader:
        skipped.extend(SECOND_READER_GROUPS)
    for value in skip_groups:
        skipped.append(str(value))
    ordered: list[str] = []
    seen: set[str] = set()
    for group in skipped:
        if group not in GROUP_ORDER:
            raise ValueError(
                f"Unknown skip group {group!r}; allowed={list(GROUP_ORDER)}"
            )
        if group in seen:
            continue
        seen.add(group)
        ordered.append(group)
    return ordered


def load_runs(
    config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    canonical_manifest: Mapping[str, Any],
    *,
    allow_incomplete: bool,
    skip_groups: Sequence[str] = (),
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    loaded: dict[str, dict[str, dict[str, Any]]] = {}
    validation: dict[str, Any] = {
        "groups": {},
        "missing_runs": [],
        "skipped_groups": list(skip_groups),
    }
    answerable_qids = [
        qid for qid in canonical_manifest["question_ids"]
        if metadata[qid]["answerability"] == "answerable"
    ]
    skip_set = set(skip_groups)
    for group in GROUP_ORDER:
        group_config = (config.get("groups") or {}).get(group) or {}
        runs_config = group_config.get("runs") or {}
        loaded[group] = {}
        validation["groups"][group] = {}
        if group in skip_set:
            validation["groups"][group] = {
                "skipped": True,
                "reason": str(
                    group_config.get("skip_reason")
                    or config.get("skip_group_reason")
                    or "skipped by skip_groups / --skip-second-reader"
                ),
            }
            continue
        for run_name, raw_config in runs_config.items():
            run_config = dict(raw_config)
            judged_path, summary_path, manifest_path = _result_paths(run_config)
            result_dir = Path(str(run_config["result_dir"]))
            inference_path = Path(str(
                run_config.get("inference_details_path")
                or result_dir / "details.jsonl"
            ))
            missing = [
                str(path) for path in (
                    inference_path, judged_path, summary_path, manifest_path
                )
                if not path.is_file()
            ]
            if missing:
                validation["missing_runs"].append({
                    "group": group, "run": run_name, "missing": missing,
                })
                if allow_incomplete:
                    continue
                raise FileNotFoundError(
                    f"Missing required paper input for {group}/{run_name}: {missing}"
                )
            label = f"{group}/{run_name}"
            try:
                raw_judged_rows = _read_jsonl(judged_path)
                rows = _decorate_rows(
                    raw_judged_rows, metadata, source=label
                )
                manifest = _read_json(manifest_path)
                if not isinstance(manifest, Mapping):
                    raise ValueError(f"Invalid run manifest: {manifest_path}")
                _validate_expected_delivery(run_config, manifest, label=label)
                _validate_stage_identity(group, run_name, manifest)
                expected_reader = group_config.get("expected_reader")
                if (
                    expected_reader is not None
                    and manifest.get("answer_reasoner") != expected_reader
                ):
                    raise ValueError(
                        f"Unexpected reader for {label}: "
                        f"{manifest.get('answer_reasoner')!r} != {expected_reader!r}"
                    )
                if manifest.get("canonical_qid_sha256") != canonical_manifest["question_id_sha256"]:
                    raise ValueError(
                        f"Canonical manifest mismatch for {group}/{run_name}"
                    )
                if int(manifest.get("evaluated_item_count", len(rows))) != len(rows):
                    raise ValueError(
                        f"Manifest/result count mismatch for {group}/{run_name}"
                    )
                if group in CONTROLLED_GROUPS and not manifest.get("config_sha256"):
                    raise ValueError(
                        f"Controlled run lacks config_sha256 for {group}/{run_name}"
                    )
                id_field = str(group_config.get(
                    "id_field",
                    "intervention_id" if group == "modality_interventions" else "question_id",
                ))
                inference_rows = _read_jsonl(inference_path)
                _validate_judged_binding(
                    inference_rows,
                    raw_judged_rows,
                    id_field=id_field,
                    label=label,
                )
                ids = _ordered_ids(rows, id_field)
                if not all(ids) or len(ids) != len(set(ids)):
                    raise ValueError(
                        f"Missing or duplicate {id_field} in {group}/{run_name}"
                    )
                population = str(group_config.get("population", "answerable"))
                expected = (
                    list(canonical_manifest["question_ids"])
                    if population == "all" else answerable_qids
                    if population == "answerable" else None
                )
                if population == "manifest":
                    population_manifest = load_qid_manifest(
                        str(group_config["qid_manifest"])
                    )
                    expected = list(population_manifest["question_ids"])
                if expected is not None and ids != expected:
                    raise ValueError(
                        f"Ordered QID population mismatch for {group}/{run_name}: "
                        f"expected={len(expected)}, actual={len(ids)}"
                    )
                expected_count = group_config.get("expected_count")
                if expected_count is not None and len(rows) != int(expected_count):
                    raise ValueError(
                        f"Expected {expected_count} rows for {group}/{run_name}, "
                        f"found {len(rows)}"
                    )
                judge_summary = _validate_judge_summary(
                    _read_json(summary_path), rows, label=label
                )
                judge_model = (
                    judge_summary.get("judge_model")
                    or _manifest_compatibility_value(manifest, "judge")
                )
                if not judge_model:
                    raise ValueError(
                        f"Judge model is unobservable for {group}/{run_name}"
                    )
                configured_judge = _manifest_compatibility_value(manifest, "judge")
                if configured_judge and str(configured_judge) != str(judge_model):
                    raise ValueError(
                        f"Configured/actual judge mismatch for {group}/{run_name}: "
                        f"{configured_judge!r} != {judge_model!r}"
                    )
                extra_paths: dict[str, Any] = {}
                if group.startswith("modality_"):
                    extra_paths = _validate_intervention_audit(
                        Path(str(run_config["result_dir"])), manifest, rows,
                        run_name=run_name,
                    )
            except (OSError, ValueError) as exc:
                validation["missing_runs"].append({
                    "group": group,
                    "run": run_name,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                if allow_incomplete:
                    continue
                raise
            run = {
                "name": run_name,
                "config": run_config,
                "rows": rows,
                "manifest": dict(manifest),
                "id_field": id_field,
                "paths": {
                    "inference_details": str(inference_path),
                    "inference_details_sha256": sha256_file(inference_path),
                    "judged_details": str(judged_path),
                    "judged_details_sha256": sha256_file(judged_path),
                    "run_manifest": str(manifest_path),
                    "run_manifest_sha256": sha256_file(manifest_path),
                    "judge_summary": str(summary_path) if summary_path.is_file() else None,
                    "judge_summary_sha256": sha256_file(summary_path)
                    if summary_path.is_file() else None,
                    **extra_paths,
                },
                "judge_model": judge_model,
            }
            loaded[group][run_name] = run
            validation["groups"][group][run_name] = {
                "num_rows": len(rows),
                "id_field": id_field,
                "ordered_id_sha256": canonical_qid_sha256(ids),
                "judge_model": run["judge_model"],
                **run["paths"],
            }

        # Compatibility is checked within comparable populations. Intervention
        # modalities have different carrier sets, so compare within each family.
        family_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run_name, run in loaded[group].items():
            family = str(run["config"].get(
                "compatibility_family",
                run["manifest"].get("modality", "all")
                if group.startswith("modality_") else "all",
            ))
            family_runs[family].append(run)
        for family, runs in family_runs.items():
            judge_models = {str(run["judge_model"]) for run in runs}
            if len(judge_models) > 1:
                raise ValueError(
                    f"Incompatible actual judge models across {group}/{family}: "
                    f"{sorted(judge_models)}"
                )
            for key in _compatible_fields(group):
                values = {
                    sha256_json(_manifest_compatibility_value(run["manifest"], key))
                    for run in runs
                }
                if len(values) > 1:
                    # Second-reader pilots were resumed over time; config_sha256
                    # can drift while reader/evidence packs stay matched. Under
                    # --allow-incomplete, warn rather than abort aggregation.
                    if (
                        allow_incomplete
                        and key == "config_sha256"
                        and "second_reader" in group
                    ):
                        validation.setdefault("warnings", []).append(
                            f"Incompatible {key} across {group}/{family} runs "
                            f"(allowed under --allow-incomplete)"
                        )
                        continue
                    raise ValueError(
                        f"Incompatible {key} across {group}/{family} runs"
                    )
            if len(runs) > 1:
                baseline = runs[0]
                for other in runs[1:]:
                    validate_pair(
                        baseline["rows"], other["rows"],
                        id_field=str(baseline["id_field"]),
                    )
        if group.startswith("stage_"):
            retrieved_hashes: dict[str, str] = {}
            for run_name, run in sorted(loaded[group].items()):
                value = run["manifest"].get("retrieval_manifest_sha256")
                if not value:
                    continue
                retrieval_path_value = run["manifest"].get("index_manifest")
                if not retrieval_path_value:
                    raise ValueError(
                        f"Missing retrieval manifest path for "
                        f"{group}/{run_name}"
                    )
                retrieval_path = Path(str(retrieval_path_value))
                if not retrieval_path.is_file():
                    raise FileNotFoundError(
                        f"Frozen retrieval manifest is unavailable for "
                        f"{group}/{run_name}: {retrieval_path}"
                    )
                if sha256_file(retrieval_path) != str(value):
                    raise ValueError(
                        f"Frozen retrieval manifest hash drift for "
                        f"{group}/{run_name}"
                    )
                retrieved_hashes[run_name] = str(value)
            if len(set(retrieved_hashes.values())) > 1:
                raise ValueError(
                    "Retrieved stage conditions use different frozen "
                    "retrieval manifests"
                )
            if retrieved_hashes:
                validation["groups"][group]["shared_retrieval_manifest"] = {
                    "sha256": next(iter(retrieved_hashes.values())),
                    "runs": sorted(retrieved_hashes),
                }
    validation["passed"] = not validation["missing_runs"]
    validation["allow_incomplete"] = allow_incomplete
    return loaded, validation


def _run_metrics(
    run: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
    label: str,
) -> dict[str, Any]:
    rows = run["rows"]
    answerable = [row for row in rows if row.get("answerability") == "answerable"]
    refusals = [row for row in rows if row.get("answerability") == "unanswerable"]
    estimate = estimate_mean(
        answerable, samples=samples, seed=seed, label=label + ":answerable"
    ) if answerable else {
        "n": 0, "mean_judge_score": None, "item_bootstrap_95_ci": None,
        "persona_clustered_bootstrap_95_ci": None, "num_personas": 0,
        "bootstrap_samples": samples,
    }
    refusal_scores = [_score(row) for row in refusals]
    latencies = [float(row.get("latency_seconds") or 0.0) for row in rows]
    costs = [
        float((row.get("usage") or {}).get("estimated_cost_usd"))
        for row in rows
        if (row.get("usage") or {}).get("estimated_cost_usd") is not None
    ]
    estimate.update({
        "num_total": len(rows),
        "num_answerable": len(answerable),
        "num_refusals": len(refusals),
        "refusal_mean_judge_score": _mean(refusal_scores),
        "correct_refusal_rate": _mean([
            float(_is_refusal(row.get("predicted_answer"))) for row in refusals
        ]),
        "false_refusal_rate": _mean([
            float(_is_refusal(row.get("predicted_answer"))) for row in answerable
        ]),
        "mean_latency_seconds": _mean(latencies),
        "total_estimated_cost_usd": round(sum(costs), 6) if costs else None,
        "judge_model": run.get("judge_model"),
    })
    return estimate


def build_breakdowns(
    loaded: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nested: dict[str, Any] = {}
    flat: list[dict[str, Any]] = []
    selectors: tuple[tuple[str, Callable[[Mapping[str, Any]], Iterable[str]]], ...] = (
        ("question_type", lambda row: [str(row.get("question_type") or "unknown")]),
        ("modality", lambda row: [str(value) for value in row.get("required_modalities", [])]),
        ("challenge", lambda row: [str(value) for value in row.get("canonical_challenges", [])]),
        ("chain_length", lambda row: [str(row.get("chain_length", 0))]),
    )
    for group, runs in loaded.items():
        nested[group] = {}
        for run_name, run in runs.items():
            nested[group][run_name] = {}
            answerable = [
                row for row in run["rows"]
                if row.get("answerability") == "answerable"
            ]
            for dimension, selector in selectors:
                buckets: dict[str, list[float]] = defaultdict(list)
                for row in answerable:
                    for value in selector(row):
                        buckets[value].append(_score(row))
                values = {
                    value: {"n": len(scores), "mean_judge_score": _mean(scores)}
                    for value, scores in sorted(buckets.items())
                }
                nested[group][run_name][dimension] = values
                for value, result in values.items():
                    flat.append({
                        "group": group,
                        "run": run_name,
                        "dimension": dimension,
                        "value": value,
                        **result,
                    })
    return nested, flat


def _comparison_rows(
    runs: Mapping[str, Mapping[str, Any]],
    specifications: Sequence[tuple[str, str, str]],
    *,
    samples: int,
    seed: int,
    group: str,
    allow_incomplete: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, left_name, right_name in specifications:
        if left_name not in runs or right_name not in runs:
            if allow_incomplete:
                continue
            raise ValueError(
                f"Comparison {name} requires {left_name} and {right_name}"
            )
        left = runs[left_name]
        right = runs[right_name]
        effect = paired_effect(
            left["rows"], right["rows"], id_field=str(left["id_field"]),
            samples=samples, seed=seed, label=f"{group}:{name}",
        )
        effect.update({"left": left_name, "right": right_name})
        output[name] = effect
        if group == "modality_interventions":
            challenges = sorted({
                str(row.get("paper_challenge") or row.get("challenge") or "unknown")
                for row in left["rows"]
            })
            by_challenge: dict[str, Any] = {}
            for challenge in challenges:
                left_rows = [
                    row for row in left["rows"]
                    if str(row.get("paper_challenge") or row.get("challenge") or "unknown") == challenge
                ]
                right_rows = [
                    row for row in right["rows"]
                    if str(row.get("paper_challenge") or row.get("challenge") or "unknown") == challenge
                ]
                by_challenge[challenge] = paired_effect(
                    left_rows, right_rows, id_field=str(left["id_field"]),
                    samples=samples, seed=seed,
                    label=f"{group}:{name}:{challenge}",
                )
            effect["by_challenge"] = by_challenge
    return output


def _aggregate_numeric(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return _mean(numeric)


def retrieval_delivery_table(
    runs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for run_name, run in runs.items():
        recalls: dict[str, list[float]] = defaultdict(list)
        chains: dict[str, list[float]] = defaultdict(list)
        delivery_chains: list[float] = []
        delivery_coverage: list[float] = []
        for row in run["rows"]:
            metric = row.get("retrieval_recall")
            metric = metric if isinstance(metric, Mapping) else {}
            trace_retrieval = (row.get("trace") or {}).get("retrieval")
            trace_retrieval = (
                trace_retrieval if isinstance(trace_retrieval, Mapping) else {}
            )
            # Backends that cannot expose ranked retrieval report the string
            # "unobservable" here; such runs contribute no recall samples.
            for key, sink in (
                ("fact_recall_at_k", recalls),
                ("complete_chain_at_k", chains),
            ):
                cutoffs = metric.get(key) or trace_retrieval.get(key)
                if not isinstance(cutoffs, Mapping):
                    continue
                for cutoff, value in cutoffs.items():
                    if isinstance(value, bool):
                        sink[str(cutoff)].append(float(value))
                    elif isinstance(value, (int, float)):
                        sink[str(cutoff)].append(float(value))
            trace_delivery = (row.get("trace") or {}).get("delivery") or {}
            value = trace_delivery.get("complete_chain_covered")
            if isinstance(value, bool):
                delivery_chains.append(float(value))
            elif isinstance(value, (int, float)):
                delivery_chains.append(float(value))
            value = trace_delivery.get("required_fact_coverage")
            if isinstance(value, (int, float)):
                delivery_coverage.append(float(value))
        output[run_name] = {
            "fact_recall_at_k": {
                cutoff: _mean(values) for cutoff, values in sorted(
                    recalls.items(), key=lambda item: int(item[0])
                )
            },
            "complete_chain_at_k": {
                cutoff: _mean(values) for cutoff, values in sorted(
                    chains.items(), key=lambda item: int(item[0])
                )
            },
            "complete_chain_delivery_rate": _mean(delivery_chains),
            "required_fact_delivery_coverage": _mean(delivery_coverage),
        }
    return output


def build_budget_curve(
    stage_runs: Mapping[str, Mapping[str, Any]],
    budget_runs: Mapping[str, Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
    allow_incomplete: bool,
) -> dict[str, Any]:
    all_delivered = stage_runs.get("retrieved_raw_all_delivered")
    if all_delivered is None:
        if allow_incomplete:
            return {}
        raise ValueError("Budget curve requires retrieved_raw_all_delivered")
    output: dict[str, Any] = {
        "all_delivered": {
            "num_units": all_delivered["manifest"].get("top_k"),
            **retrieval_delivery_table(
                {"all_delivered": all_delivered}
            )["all_delivered"],
        },
        "budget_points": {},
    }
    for name, run in budget_runs.items():
        delivery_budget = run["manifest"].get("delivery_budget") or {}
        effect = paired_effect(
            all_delivered["rows"],
            run["rows"],
            id_field=str(run["id_field"]),
            samples=samples,
            seed=seed,
            label=f"stage_budget_sweep:{name}",
        )
        delivery = retrieval_delivery_table({name: run})[name]
        output["budget_points"][name] = {
            "production_top_k": delivery_budget.get("production_top_k"),
            "policy": delivery_budget.get("policy"),
            "delivery_loss": effect,
            **delivery,
        }
    return output


def build_reader_replication(
    primary_runs: Mapping[str, Mapping[str, Any]],
    secondary_runs: Mapping[str, Mapping[str, Any]],
    specifications: Sequence[tuple[str, str, str]],
    *,
    samples: int,
    seed: int,
    label: str,
    allow_incomplete: bool,
) -> dict[str, Any]:
    if not primary_runs or not secondary_runs:
        return {
            "status": "deferred",
            "reason": (
                "Second-reader replication skipped or incomplete "
                f"({label}: primary_runs={len(primary_runs)}, "
                f"secondary_runs={len(secondary_runs)})"
            ),
            "contrasts": {},
            "num_shared_contrasts": 0,
            "sign_agreement_count": 0,
            "sign_agreement_rate": None,
            "contrast_rank_spearman": None,
            "primary_perception_cost": None,
            "secondary_perception_cost": None,
            "evidence_validation": {},
        }
    evidence_validation: dict[str, Any] = {}
    for condition in sorted(set(primary_runs) & set(secondary_runs)):
        primary_run = primary_runs[condition]
        secondary_run = secondary_runs[condition]
        for field in ("answer_prompt_version",):
            primary_value = primary_run["manifest"].get(field)
            secondary_value = secondary_run["manifest"].get(field)
            if not primary_value or primary_value != secondary_value:
                raise ValueError(
                    f"Reader replication changed {field} for "
                    f"{label}/{condition}: {primary_value!r} != "
                    f"{secondary_value!r}"
                )
        if primary_run.get("judge_model") != secondary_run.get("judge_model"):
            raise ValueError(
                f"Reader replication changed judge for {label}/{condition}: "
                f"{primary_run.get('judge_model')!r} != "
                f"{secondary_run.get('judge_model')!r}"
            )
        id_field = str(primary_run["id_field"])
        identifiers = validate_pair(
            primary_run["rows"], secondary_run["rows"], id_field=id_field
        )
        primary_by_id = {
            str(row[id_field]): row for row in primary_run["rows"]
        }
        secondary_by_id = {
            str(row[id_field]): row for row in secondary_run["rows"]
        }
        mismatches: list[str] = []
        for identifier in identifiers:
            signatures = []
            for row in (
                primary_by_id[identifier], secondary_by_id[identifier]
            ):
                trace = row.get("trace") or {}
                if not isinstance(trace.get("retrieval"), Mapping) or not isinstance(
                    trace.get("delivery"), Mapping
                ):
                    raise ValueError(
                        f"Reader replication lacks evidence trace for "
                        f"{label}/{condition}/{identifier}"
                    )
                signatures.append(sha256_json({
                    "query": trace.get("query"),
                    "retrieval_ranked": (
                        (trace.get("retrieval") or {}).get("ranked")
                    ),
                    "delivery": {
                        key: (trace.get("delivery") or {}).get(key)
                        for key in ("delivered", "dropped", "failed")
                    },
                    "leak_audit": {
                        key: (row.get("leak_audit") or {}).get(key)
                        for key in (
                            "baseline_target_signature",
                            "observed_target_signature",
                            "baseline_non_target_signature",
                            "observed_non_target_signature",
                        )
                    },
                }))
            if signatures[0] != signatures[1]:
                mismatches.append(identifier)
        if mismatches:
            raise ValueError(
                f"Reader replication evidence differs for {label}/{condition}: "
                f"{mismatches[:5]}"
            )
        evidence_validation[condition] = {
            "num_items": len(identifiers),
            "byte_identical_trace_evidence": True,
        }
    primary = _comparison_rows(
        primary_runs,
        specifications,
        samples=samples,
        seed=seed,
        group=f"{label}:primary",
        allow_incomplete=allow_incomplete,
    )
    secondary = _comparison_rows(
        secondary_runs,
        specifications,
        samples=samples,
        seed=seed,
        group=f"{label}:secondary",
        allow_incomplete=allow_incomplete,
    )
    shared = [name for name, _, _ in specifications if name in primary and name in secondary]
    primary_effects = [
        float(primary[name]["mean_paired_effect_left_minus_right"])
        for name in shared
    ]
    secondary_effects = [
        float(secondary[name]["mean_paired_effect_left_minus_right"])
        for name in shared
    ]
    sign_matches = [
        (left == 0.0 and right == 0.0) or (left * right > 0.0)
        for left, right in zip(primary_effects, secondary_effects)
    ]
    return {
        "contrasts": {
            name: {
                "primary": primary[name],
                "secondary": secondary[name],
                "sign_agreement": sign_matches[index],
            }
            for index, name in enumerate(shared)
        },
        "num_shared_contrasts": len(shared),
        "sign_agreement_count": sum(sign_matches),
        "sign_agreement_rate": _mean([float(value) for value in sign_matches]),
        "contrast_rank_spearman": _pearson(
            _rankdata(primary_effects), _rankdata(secondary_effects)
        ) if len(shared) >= 2 else None,
        "primary_perception_cost": (
            primary.get("perception_cost", {}).get(
                "mean_paired_effect_left_minus_right"
            )
        ),
        "secondary_perception_cost": (
            secondary.get("perception_cost", {}).get(
                "mean_paired_effect_left_minus_right"
            )
        ),
        "evidence_validation": evidence_validation,
    }


def stage_funnel(
    stage_metrics: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    conditions = {
        name: metrics.get("mean_judge_score")
        for name, metrics in stage_metrics.items()
    }
    symbolic = conditions.get("gold_symbolic_facts")
    return {
        "condition_scores": conditions,
        "diagnostic_losses": {
            name: value.get("mean_paired_effect_left_minus_right")
            for name, value in comparisons.items()
        },
        "residual_read_reasoning_error": (
            round(1.0 - float(symbolic), 6) if symbolic is not None else None
        ),
        "interpretation": {
            "perception": "gold_symbolic_facts - gold_raw_evidence",
            "write_compression": "gold_structured_text - gold_generic_text",
            "retrieval": "gold_raw_evidence - retrieved_raw_all_delivered",
            "delivery": "retrieved_raw_all_delivered - retrieved_raw_budgeted",
            "production_query": "retrieved_raw_budgeted - end_to_end",
            "read_reasoning": "1 - gold_symbolic_facts",
        },
    }


def _rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return round(numerator / denominator, 6) if denominator else None


def build_optional_robustness(
    loaded: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    secondary_system_scores: list[tuple[str, float, float]] = []
    output: dict[str, Any] = {"runs": {}}
    for group, runs in loaded.items():
        for run_name, run in runs.items():
            run_output: dict[str, Any] = {}
            for kind, config_key in (
                ("secondary_judge", "secondary_judged_details_path"),
                ("repeated_run", "repeat_judged_details_path"),
            ):
                path_value = run["config"].get(config_key)
                if not path_value:
                    continue
                path = Path(str(path_value))
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Configured {kind} input is missing: {path}"
                    )
                other = _decorate_rows(_read_jsonl(path), metadata)
                primary_rows = list(run["rows"])
                if kind == "repeated_run":
                    id_field = str(run["id_field"])
                    other_ids = _ordered_ids(other, id_field)
                    if not all(other_ids) or len(other_ids) != len(set(other_ids)):
                        raise ValueError(
                            f"Repeated run has missing or duplicate {id_field}: {path}"
                        )
                    primary_by_id = {
                        str(row.get(id_field) or ""): row for row in primary_rows
                    }
                    missing = [value for value in other_ids if value not in primary_by_id]
                    if missing:
                        raise ValueError(
                            f"Repeated subset contains unavailable IDs: {missing[:5]}"
                        )
                    expected_order = [
                        identifier
                        for identifier in _ordered_ids(primary_rows, id_field)
                        if identifier in set(other_ids)
                    ]
                    if other_ids != expected_order:
                        raise ValueError(
                            f"Repeated subset order differs from primary run: {path}"
                        )
                    primary_rows = [primary_by_id[identifier] for identifier in other_ids]
                else:
                    validate_pair(
                        primary_rows, other, id_field=str(run["id_field"])
                    )
                left_scores = [_score(row) for row in primary_rows]
                right_scores = [_score(row) for row in other]
                result = {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "n": len(other),
                    "primary_mean": _mean(left_scores),
                    "comparison_mean": _mean(right_scores),
                    "mean_difference": _mean([
                        a - b for a, b in zip(left_scores, right_scores)
                    ]),
                    "item_score_pearson": _pearson(left_scores, right_scores),
                    "item_score_spearman": _pearson(
                        _rankdata(left_scores), _rankdata(right_scores)
                    ),
                    "answer_change_rate": _mean([
                        float(str(a.get("predicted_answer") or "").strip()
                              != str(b.get("predicted_answer") or "").strip())
                        for a, b in zip(primary_rows, other)
                    ]),
                }
                run_output[kind] = result
                if kind == "secondary_judge":
                    secondary_system_scores.append((
                        f"{group}/{run_name}",
                        float(result["primary_mean"]),
                        float(result["comparison_mean"]),
                    ))
            if run_output:
                output["runs"][f"{group}/{run_name}"] = run_output
    if secondary_system_scores:
        primary = [value[1] for value in secondary_system_scores]
        secondary = [value[2] for value in secondary_system_scores]
        output["secondary_judge_system_level"] = {
            "n_systems": len(secondary_system_scores),
            "systems": [value[0] for value in secondary_system_scores],
            "pearson": _pearson(primary, secondary),
            "spearman": _pearson(_rankdata(primary), _rankdata(secondary)),
        }
    return output


def _format_ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return "--"
    return f"[{value[0]:.3f}, {value[1]:.3f}]"


def build_markdown(
    metrics: Mapping[str, Mapping[str, Mapping[str, Any]]],
    stage_comparisons: Mapping[str, Mapping[str, Any]],
    intervention_comparisons: Mapping[str, Mapping[str, Any]],
    fixed_comparisons: Mapping[str, Mapping[str, Any]],
) -> str:
    lines = ["# MOSAIC Paper Results", ""]
    for title, group in (
        ("End-to-End Capability", "end_to_end"),
        ("Stage Conditions", "stage_decomposition"),
        ("Modality Interventions", "modality_interventions"),
        ("Fixed-Reader Architectures", "fixed_reader"),
    ):
        lines.extend([
            f"## {title}", "",
            "| Run | n | Answerable judge | Item 95% CI | Persona 95% CI | Correct refusal | False refusal |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name, value in metrics.get(group, {}).items():
            mean = value.get("mean_judge_score")
            lines.append(
                f"| {name} | {value.get('num_total', value.get('n', 0))} | "
                f"{mean:.3f} | {_format_ci(value.get('item_bootstrap_95_ci'))} | "
                f"{_format_ci(value.get('persona_clustered_bootstrap_95_ci'))} | "
                f"{value.get('correct_refusal_rate') if value.get('correct_refusal_rate') is not None else '--'} | "
                f"{value.get('false_refusal_rate') if value.get('false_refusal_rate') is not None else '--'} |"
                if mean is not None else
                f"| {name} | {value.get('num_total', 0)} | -- | -- | -- | -- | -- |"
            )
        lines.append("")
    for title, comparisons in (
        ("Stage Paired Effects", stage_comparisons),
        ("Intervention Paired Effects", intervention_comparisons),
        ("Fixed-Reader Paired Effects", fixed_comparisons),
    ):
        lines.extend([
            f"## {title}", "",
            "Positive effects favor the left condition.", "",
            "| Effect | Left | Right | n | Mean effect | Item 95% CI | Persona 95% CI | W/T/L |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for name, value in comparisons.items():
            lines.append(
                f"| {name} | {value.get('left')} | {value.get('right')} | "
                f"{value.get('n')} | {value.get('mean_paired_effect_left_minus_right'):.3f} | "
                f"{_format_ci(value.get('item_bootstrap_95_ci'))} | "
                f"{_format_ci(value.get('persona_clustered_bootstrap_95_ci'))} | "
                f"{value.get('wins')}/{value.get('ties')}/{value.get('losses')} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EVALUATION_ROOT / "configs" / "paper_results.json"),
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Process completed runs while recording missing required inputs",
    )
    parser.add_argument(
        "--skip-second-reader",
        action="store_true",
        help=(
            "Skip E2/E3 second-reader replication groups "
            "(MiMo second reader + Gemini pilot pair). "
            "Main stage/modality/fixed-reader tables still build."
        ),
    )
    parser.add_argument(
        "--skip-groups",
        action="append",
        default=[],
        help=(
            "Skip a paper_results group by name. Repeatable. "
            f"Known groups: {', '.join(GROUP_ORDER)}"
        ),
    )
    parser.add_argument(
        "--bootstrap-samples", type=int,
        help="Override the configured bootstrap replicate count",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = int(args.bootstrap_samples or config.get("bootstrap_samples", 10000))
    if samples < 100:
        raise ValueError("Use at least 100 bootstrap samples")
    seed = int(config.get("seed", 64))
    skip_groups = _resolve_skip_groups(
        config,
        skip_second_reader=args.skip_second_reader,
        skip_groups=args.skip_groups,
    )

    canonical_manifest_path = config.get(
        "canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"
    )
    canonical_manifest = load_qid_manifest(canonical_manifest_path)
    records = load_canonical_records(config.get("canonical_data_dir"))
    metadata = canonical_metadata(records)
    if list(metadata) != list(canonical_manifest["question_ids"]):
        raise ValueError("Canonical data order differs from canonical QID manifest")

    loaded, validation = load_runs(
        config, metadata, canonical_manifest,
        allow_incomplete=args.allow_incomplete,
        skip_groups=skip_groups,
    )
    _write_json(output_dir / "manifest_validation.json", validation)

    metrics: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for group, runs in loaded.items():
        metrics[group] = {}
        for run_name, run in runs.items():
            value = _run_metrics(
                run, samples=samples, seed=seed, label=f"{group}:{run_name}"
            )
            value.update({
                key: run["config"].get(key)
                for key in ("label", "system", "representation", "reader")
                if run["config"].get(key) is not None
            })
            metrics[group][run_name] = value
            metric_rows.append({"group": group, "run": run_name, **value})

    stage_comparisons = _comparison_rows(
        loaded["stage_decomposition"], STAGE_COMPARISONS,
        samples=samples, seed=seed, group="stage_decomposition",
        allow_incomplete=args.allow_incomplete,
    )
    intervention_comparisons = _comparison_rows(
        loaded["modality_interventions"], INTERVENTION_COMPARISONS,
        samples=samples, seed=seed, group="modality_interventions",
        allow_incomplete=args.allow_incomplete,
    )
    fixed_comparisons = _comparison_rows(
        loaded["fixed_reader"], FIXED_READER_COMPARISONS,
        samples=samples, seed=seed, group="fixed_reader",
        allow_incomplete=args.allow_incomplete,
    )
    breakdowns, breakdown_rows = build_breakdowns(loaded)
    retrieval_delivery = {
        "stage_decomposition": retrieval_delivery_table(
            loaded["stage_decomposition"]
        ),
        "fixed_reader": retrieval_delivery_table(
            loaded.get("fixed_reader") or {}
        ),
    }
    budget_curve = build_budget_curve(
        loaded["stage_decomposition"],
        loaded["stage_budget_sweep"],
        samples=samples,
        seed=seed,
        allow_incomplete=args.allow_incomplete,
    )
    reader_replication = {
        "stage_decomposition": build_reader_replication(
            loaded["stage_second_reader_mimo"],
            loaded["stage_second_reader_gemini"],
            STAGE_COMPARISONS,
            samples=samples,
            seed=seed,
            label="stage_second_reader",
            allow_incomplete=args.allow_incomplete,
        ),
        "modality_interventions": build_reader_replication(
            loaded["modality_second_reader_mimo"],
            loaded["modality_second_reader_gemini"],
            INTERVENTION_COMPARISONS,
            samples=samples,
            seed=seed,
            label="modality_second_reader",
            allow_incomplete=args.allow_incomplete,
        ),
    }
    funnel = stage_funnel(
        metrics["stage_decomposition"], stage_comparisons
    )
    robustness = build_optional_robustness(loaded, metadata)

    provenance = {
        "builder_version": 1,
        "config_path": os.path.abspath(args.config),
        "config_sha256": sha256_file(args.config),
        "canonical_qid_manifest": os.path.abspath(canonical_manifest_path),
        "canonical_qid_manifest_sha256": sha256_file(canonical_manifest_path),
        "canonical_qid_sha256": canonical_manifest["question_id_sha256"],
        "bootstrap_samples": samples,
        "seed": seed,
        "allow_incomplete": args.allow_incomplete,
        "skip_groups": skip_groups,
        "input_manifests": {
            f"{group}/{name}": run["paths"]
            for group, runs in loaded.items()
            for name, run in runs.items()
        },
    }
    payload = {
        "provenance": provenance,
        "manifest_validation": validation,
        "run_metrics": metrics,
        "stage_comparisons": stage_comparisons,
        "modality_intervention_comparisons": intervention_comparisons,
        "fixed_reader_comparisons": fixed_comparisons,
        "stage_failure_funnel": funnel,
        "retrieval_and_delivery": retrieval_delivery,
        "delivery_budget_curve": budget_curve,
        "reader_replication": reader_replication,
        "breakdowns": breakdowns,
        "robustness": robustness,
    }
    _write_json(output_dir / "paper_results.json", payload)
    _write_json(output_dir / "stage_failure_funnel.json", funnel)
    _write_json(output_dir / "retrieval_and_delivery.json", retrieval_delivery)
    _write_json(output_dir / "delivery_budget_curve.json", budget_curve)
    _write_json(output_dir / "reader_replication.json", reader_replication)
    _write_json(output_dir / "breakdowns.json", breakdowns)
    _write_json(output_dir / "robustness.json", robustness)
    _write_csv(output_dir / "run_metrics.csv", metric_rows)
    _write_csv(output_dir / "breakdowns.csv", breakdown_rows)

    comparison_rows: list[dict[str, Any]] = []
    for group, comparisons in (
        ("stage_decomposition", stage_comparisons),
        ("modality_interventions", intervention_comparisons),
        ("fixed_reader", fixed_comparisons),
    ):
        for name, value in comparisons.items():
            comparison_rows.append({
                "group": group, "comparison": name,
                **{key: result for key, result in value.items() if key != "by_challenge"},
            })
            for challenge, result in (value.get("by_challenge") or {}).items():
                comparison_rows.append({
                    "group": group, "comparison": name,
                    "challenge": challenge, **result,
                })
    _write_csv(output_dir / "paired_effects.csv", comparison_rows)
    (output_dir / "paper_tables.md").write_text(
        build_markdown(
            metrics, stage_comparisons,
            intervention_comparisons, fixed_comparisons,
        ),
        encoding="utf-8",
    )
    print(f"Paper results written to: {output_dir}")
    print(f"Loaded runs: {sum(len(runs) for runs in loaded.values())}")
    if skip_groups:
        print(f"Skipped groups: {', '.join(skip_groups)}")
    if validation["missing_runs"]:
        print(f"Missing runs recorded: {len(validation['missing_runs'])}")


if __name__ == "__main__":
    main()
