"""Shared evaluation protocol primitives.

This module contains the parts of the evaluation contract that must not vary
between backends: the canonical question view, provenance hashes, retrieval
metrics, evidence delivery decisions, and per-run/per-item trace metadata.
It deliberately has no model or index dependencies, so it can be used by
sanity checks and by evaluators before optional ML packages are imported.
"""

from __future__ import annotations

import hashlib
import json
import base64
import mimetypes
import os
import subprocess
import threading
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CANONICAL_ITEM_GLOB = "stage2_with_media_*.json"
def _data_root_candidate(relative: str) -> str:
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        return ""
    return f"{root}/{relative.lstrip('/')}"


DEFAULT_CANONICAL_ROOTS = (
    os.environ.get("MOSAIC_CANONICAL_DATA_DIR", ""),
    _data_root_candidate(
        "human_filtered_evaluation_results/dataset/stage2_human_filtered"
    ),
    "human_filtered_evaluation_results/dataset/stage2_human_filtered",
)
DEFAULT_QID_MANIFEST = "evaluation/manifests/canonical_qids.json"
DROP_REASONS = {
    "media_count_cap",
    "media_size_cap",
    "provider_unsupported",
    "context_token_cap",
    "observation_truncation",
    "serialization_error",
    "missing_file",
}
MEDIA_MODALITIES = {"image", "audio", "video"}


def _json_safe(value: Any) -> Any:
    """Convert values that ``json.dumps`` cannot encode (torch/numpy dtypes)."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    # torch.dtype, numpy.dtype, Path, etc.
    type_name = type(value).__name__
    if type_name == "dtype" or value.__class__.__module__.startswith(
        ("torch", "numpy")
    ):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(_json_bytes(value))


def resolve_canonical_root(root: str | os.PathLike[str] | None = None) -> str:
    """Resolve the canonical 923-item directory without copying its media."""
    candidates = (str(root),) if root else DEFAULT_CANONICAL_ROOTS
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            files = list(Path(candidate).glob(CANONICAL_ITEM_GLOB))
            if files:
                return os.path.abspath(candidate)
    searched = ", ".join(repr(x) for x in candidates if x)
    raise FileNotFoundError(
        "Canonical dataset directory was not found. Searched: " + searched
    )


def canonical_files(root: str | os.PathLike[str] | None = None) -> list[str]:
    resolved = resolve_canonical_root(root)
    return [str(path) for path in sorted(Path(resolved).glob(CANONICAL_ITEM_GLOB))]


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def load_canonical_records(
    root: str | os.PathLike[str] | None = None,
    *,
    validate: bool = True,
) -> list[dict[str, Any]]:
    """Load one flattened record per canonical question.

    The returned record keeps the original item under ``item`` and adds the
    persona UUID/name and source filename.  This is an overlay view: callers
    should use their existing index/store for memory units, never these files
    as a replacement index.
    """
    records: list[dict[str, Any]] = []
    for filename in canonical_files(root):
        payload = _load_json(filename)
        persona = payload.get("persona") or {}
        persona_uuid = str(persona.get("uuid") or "")
        for item_index, item in enumerate(payload.get("items") or []):
            if not isinstance(item, dict):
                raise ValueError(f"Non-object item at {filename}:{item_index}")
            question_id = str(item.get("question_id") or "")
            if not question_id:
                raise ValueError(f"Missing question_id at {filename}:{item_index}")
            records.append(
                {
                    "question_id": question_id,
                    "persona_uuid": persona_uuid,
                    "persona_name": persona.get("name", ""),
                    "question_type": item.get("question_type", "unknown"),
                    "answerability": item.get("answerability", "unknown"),
                    "source_filename": os.path.basename(filename),
                    "source_path": os.path.abspath(filename),
                    "item_index": item_index,
                    "item": item,
                }
            )

    if validate:
        validate_canonical_records(records)
    return records


def validate_canonical_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_items: int = 923,
    expected_answerable: int = 784,
    expected_refusals: int = 139,
    expected_personas: int = 70,
) -> dict[str, Any]:
    """Validate the canonical identity/count invariants."""
    qids = [str(record.get("question_id") or "") for record in records]
    duplicates = sorted(qid for qid, count in Counter(qids).items() if count > 1)
    missing_ids = [qid for qid in qids if not qid]
    answerable = sum(record.get("answerability") == "answerable" for record in records)
    refusals = sum(record.get("answerability") == "unanswerable" for record in records)
    personas = {str(record.get("persona_uuid") or "") for record in records}
    stats = {
        "num_items": len(records),
        "num_unique_question_ids": len(set(qids)),
        "duplicate_question_ids": duplicates,
        "missing_question_ids": missing_ids,
        "num_answerable": answerable,
        "num_refusals": refusals,
        "num_personas": len(personas - {""}),
    }
    failures: list[str] = []
    if expected_items is not None and len(records) != expected_items:
        failures.append(f"expected {expected_items} items, found {len(records)}")
    if duplicates:
        failures.append(f"duplicate question IDs: {duplicates[:5]}")
    if missing_ids:
        failures.append(f"missing question IDs: {len(missing_ids)}")
    if expected_answerable is not None and answerable != expected_answerable:
        failures.append(f"expected {expected_answerable} answerable, found {answerable}")
    if expected_refusals is not None and refusals != expected_refusals:
        failures.append(f"expected {expected_refusals} refusals, found {refusals}")
    if expected_personas is not None and len(personas - {""}) != expected_personas:
        failures.append(
            f"expected {expected_personas} personas, found {len(personas - {''})}"
        )
    if failures:
        raise ValueError("Canonical dataset validation failed: " + "; ".join(failures))
    return stats


def canonical_qid_sha256(records_or_qids: Iterable[Any]) -> str:
    """Hash the ordered QID list, one ID per line."""
    qids = [
        str(value.get("question_id") if isinstance(value, Mapping) else value)
        for value in records_or_qids
    ]
    return sha256_bytes(("\n".join(qids) + "\n").encode("utf-8"))


def build_qid_manifest(
    root: str | os.PathLike[str] | None = None,
    output_path: str | os.PathLike[str] = DEFAULT_QID_MANIFEST,
) -> dict[str, Any]:
    """Create the small repository-local canonical QID manifest."""
    resolved = resolve_canonical_root(root)
    records = load_canonical_records(resolved, validate=True)
    entries = [
        {
            key: record[key]
            for key in (
                "question_id",
                "persona_uuid",
                "persona_name",
                "question_type",
                "answerability",
                "source_filename",
                "item_index",
            )
        }
        for record in records
    ]
    manifest = {
        "manifest_version": 1,
        "dataset_root": resolved,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_items": len(entries),
        "num_answerable": sum(e["answerability"] == "answerable" for e in entries),
        "num_refusals": sum(e["answerability"] == "unanswerable" for e in entries),
        "num_personas": len({e["persona_uuid"] for e in entries}),
        "question_id_sha256": canonical_qid_sha256(entries),
        "question_ids": [e["question_id"] for e in entries],
        "items": entries,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def load_qid_manifest(path: str | os.PathLike[str] = DEFAULT_QID_MANIFEST) -> dict[str, Any]:
    manifest = _load_json(str(path))
    ids = manifest.get("question_ids") or [x.get("question_id") for x in manifest.get("items", [])]
    if not ids or canonical_qid_sha256(ids) != manifest.get("question_id_sha256"):
        raise ValueError(f"Invalid or tampered QID manifest: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"QID manifest contains duplicate IDs: {path}")
    return manifest


def records_by_persona(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("persona_uuid") or "")].append(dict(record))
    return dict(grouped)


class QueryManifestCache:
    """Thread-safe cached retrieval queries for controlled comparisons."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None,
        *,
        policy: str = "original",
        rewrite_model: str | None = None,
    ) -> None:
        if policy not in {"original", "fixed_rewrite", "adaptive_react"}:
            raise ValueError(f"Unknown query policy: {policy}")
        self.path = str(path) if path else None
        self.policy = policy
        self.rewrite_model = rewrite_model
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if self.path and os.path.exists(self.path):
            payload = _load_json(self.path)
            entries = payload.get("queries", payload)
            if not isinstance(entries, Mapping):
                raise ValueError(f"Invalid query manifest: {self.path}")
            self._entries = {str(key): dict(value) for key, value in entries.items()}

    def resolve(
        self,
        question_id: str,
        original: str,
        rewrite_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Return original/retrieval text and cache a fixed rewrite once."""
        if self.policy in {"original", "adaptive_react"}:
            return {
                "original": original,
                "retrieval_query": original,
                "rewrite_model": None,
                "query_policy": self.policy,
            }
        with self._lock:
            cached = self._entries.get(question_id)
            if cached:
                if cached.get("original") not in {None, original}:
                    raise ValueError(
                        f"Cached query {question_id} has a different original question"
                    )
                return dict(cached)
            if rewrite_fn is None:
                raise ValueError(
                    f"No cached fixed rewrite for {question_id} and no rewrite function was supplied"
                )
            rewritten = str(rewrite_fn(original) or original).strip() or original
            entry = {
                "original": original,
                "retrieval_query": rewritten,
                "rewrite_model": self.rewrite_model,
                "query_policy": self.policy,
            }
            self._entries[question_id] = entry
            self._save_locked()
            return dict(entry)

    def _save_locked(self) -> None:
        if not self.path:
            return
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest_version": 1,
            "query_policy": self.policy,
            "rewrite_model": self.rewrite_model,
            "num_queries": len(self._entries),
            "queries": dict(sorted(self._entries.items())),
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, destination)

    @property
    def entries(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: dict(value) for key, value in self._entries.items()}


def overlay_persona_items(
    index_items: Sequence[Mapping[str, Any]],
    canonical_items: Sequence[Mapping[str, Any]],
    *,
    only_qids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select canonical question metadata while retaining index-item order only as a fallback.

    ``index_items`` is intentionally not used as the source of truth for gold
    answers.  It is accepted to make adapter integration straightforward and
    to support old stores whose item order differs from the canonical view.
    """
    canonical_by_qid = {
        str(record.get("question_id")): dict(record.get("item") or record)
        for record in canonical_items
    }
    index_qids = [str(item.get("question_id") or "") for item in index_items]
    ordered_ids = [qid for qid in index_qids if qid in canonical_by_qid]
    canonical_ids = [str(record.get("question_id")) for record in canonical_items]
    for qid in canonical_ids:
        if qid not in ordered_ids:
            ordered_ids.append(qid)
    if only_qids is not None:
        ordered_ids = [qid for qid in ordered_ids if qid in only_qids]
    return [canonical_by_qid[qid] for qid in ordered_ids]


def validate_result_qids(
    results: Sequence[Mapping[str, Any]],
    expected_qids: Sequence[str],
) -> dict[str, Any]:
    """Fail on missing, extra, duplicate, or differently ordered result IDs."""
    actual = [
        str(result.get("question_id") or result.get("item_id") or "")
        for result in results
    ]
    expected = [str(qid) for qid in expected_qids]
    duplicates = sorted(qid for qid, count in Counter(actual).items() if count > 1)
    missing = [qid for qid in expected if qid not in set(actual)]
    extra = [qid for qid in actual if qid not in set(expected)]
    if duplicates or missing or extra or actual != expected:
        raise ValueError(
            "Result QID validation failed: "
            f"duplicates={len(duplicates)}, missing={len(missing)}, "
            f"extra={len(extra)}, ordered={actual == expected}"
        )
    return {
        "num_results": len(actual),
        "num_expected": len(expected),
        "ordered": True,
        "question_id_sha256": canonical_qid_sha256(actual),
    }


def _fact_keys(unit: Mapping[str, Any]) -> set[str]:
    values: list[Any] = [unit.get("fact_key"), unit.get("fact_keys")]
    keys: set[str] = set()
    for value in values:
        if isinstance(value, str) and value:
            keys.add(value)
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, str, Mapping)):
            keys.update(str(v) for v in value if v)
    return keys


def _session_id(unit: Mapping[str, Any]) -> str | None:
    value = unit.get("session_id") or unit.get("source_session_id")
    return str(value) if value else None


def _required_fact_info(evidence_plan: Sequence[Mapping[str, Any]]) -> tuple[set[str], dict[str, str]]:
    required: set[str] = set()
    fact_sessions: dict[str, str] = {}
    for evidence in evidence_plan:
        if not evidence.get("required_for_answer"):
            continue
        fact_key = evidence.get("fact_key")
        if not fact_key:
            continue
        key = str(fact_key)
        required.add(key)
        session = (
            evidence.get("session_id")
            or evidence.get("source_session_id")
            or evidence.get("answer_session_id")
        )
        if session:
            fact_sessions[key] = str(session)
    return required, fact_sessions


def compute_retrieval_recall(
    retrieved: Sequence[Mapping[str, Any]],
    evidence_plan: Sequence[Mapping[str, Any]],
    k_values: Sequence[int] = (1, 3, 5, 10, 20),
    *,
    required_session_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute rank-sensitive fact/session recall and complete-chain coverage.

    Facts are unioned *within* ``retrieved[:k]``.  A fact at rank 20 therefore
    cannot contribute to R@1, R@3, or R@10, even when another duplicate appears
    later in the list.
    """
    required, fact_sessions = _required_fact_info(evidence_plan)
    explicit_sessions = {str(x) for x in (required_session_ids or ()) if x}
    required_sessions = explicit_sessions | set(fact_sessions.values())
    if not required:
        result = {
            "status": "not_applicable",
            "recall": {f"recall@{k}": "not_applicable" for k in k_values},
            "fact_recall_at_k": {str(k): "not_applicable" for k in k_values},
            "complete_chain_at_k": {str(k): "not_applicable" for k in k_values},
            "required_session_recall_at_k": {str(k): "not_applicable" for k in k_values},
            "num_required": 0,
            "required_fact_keys": [],
            "required_session_ids": sorted(required_sessions),
            "found_fact_keys": [],
        }
        return result

    fact_recall: dict[str, float] = {}
    complete_chain: dict[str, bool] = {}
    session_recall: dict[str, float | str] = {}
    found_union: set[str] = set()
    for cutoff in k_values:
        prefix = retrieved[: max(0, int(cutoff))]
        facts_at_k: set[str] = set()
        sessions_at_k: set[str] = set()
        for unit in prefix:
            facts_at_k.update(_fact_keys(unit))
            session_id = _session_id(unit)
            if session_id:
                sessions_at_k.add(session_id)
        found_union.update(facts_at_k & required)
        key = str(cutoff)
        fact_recall[key] = round(len(facts_at_k & required) / len(required), 4)
        complete_chain[key] = required.issubset(facts_at_k)
        if required_sessions:
            session_recall[key] = round(
                len(sessions_at_k & required_sessions) / len(required_sessions), 4
            )
        else:
            session_recall[key] = "not_applicable"

    return {
        "status": "ok",
        "recall": {f"recall@{k}": fact_recall[str(k)] for k in k_values},
        "fact_recall_at_k": fact_recall,
        "complete_chain_at_k": complete_chain,
        "required_session_recall_at_k": session_recall,
        "num_required": len(required),
        "required_fact_keys": sorted(required),
        "required_session_ids": sorted(required_sessions),
        "found_fact_keys": sorted(found_union),
    }


def retrieval_metric_aggregate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate retrieval metrics with refusal/empty-chain exclusions."""
    eligible = [
        result for result in results
        if result.get("answerability") != "unanswerable"
        and result.get("retrieval_recall", {}).get("status") != "not_applicable"
        and result.get("retrieval_recall", {}).get("num_required", 0) > 0
    ]
    cutoffs = sorted(
        {
            key
            for result in eligible
            for key in result.get("retrieval_recall", {}).get("fact_recall_at_k", {})
        },
        key=lambda value: int(value),
    )
    macro: dict[str, float] = {}
    complete: dict[str, float] = {}
    micro: dict[str, float] = {}
    for cutoff in cutoffs:
        recalls = [
            result["retrieval_recall"]["fact_recall_at_k"][cutoff]
            for result in eligible
            if isinstance(result["retrieval_recall"]["fact_recall_at_k"].get(cutoff), (int, float))
        ]
        chains = [
            result["retrieval_recall"]["complete_chain_at_k"][cutoff]
            for result in eligible
            if isinstance(result["retrieval_recall"]["complete_chain_at_k"].get(cutoff), bool)
        ]
        numerator = 0
        denominator = 0
        for result in eligible:
            metric = result["retrieval_recall"]
            value = metric.get("fact_recall_at_k", {}).get(cutoff)
            if isinstance(value, (int, float)):
                numerator += round(value * metric["num_required"])
                denominator += metric["num_required"]
        macro[f"recall@{cutoff}"] = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
        complete[f"complete_chain@{cutoff}"] = round(sum(chains) / len(chains), 4) if chains else 0.0
        micro[f"recall@{cutoff}"] = round(numerator / denominator, 4) if denominator else 0.0
    return {
        "num_items_with_required_facts": len(eligible),
        "macro_fact_recall": macro,
        "micro_fact_recall": micro,
        "complete_chain_rate": complete,
    }


def aggregate_results(item_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate answer outcomes while keeping answerability strata separate."""
    answerable = [r for r in item_results if r.get("answerability") == "answerable"]
    refusals = [r for r in item_results if r.get("answerability") == "unanswerable"]
    correct_answerable = sum(
        bool(r.get("evaluation", {}).get("alias_match")) for r in answerable
    )
    correct_refusals = sum(
        bool(r.get("evaluation", {}).get("unanswerable_correct")) for r in refusals
    )
    false_refusals = sum(
        bool(r.get("evaluation", {}).get("unanswerable_correct"))
        or str(r.get("predicted_answer", "")).strip().lower() in {"i don't know", "i don't know."}
        for r in answerable
    )
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in item_results:
        by_type[str(result.get("question_type", "unknown"))].append(result)
    per_type: dict[str, Any] = {}
    for question_type, rows in sorted(by_type.items()):
        correct = sum(
            bool(row.get("evaluation", {}).get("alias_match"))
            if row.get("answerability") == "answerable"
            else bool(row.get("evaluation", {}).get("unanswerable_correct"))
            for row in rows
        )
        per_type[question_type] = {
            "count": len(rows),
            "mean_match_score": round(correct / len(rows), 4) if rows else 0.0,
        }
    total = len(item_results)
    return {
        "total_questions": total,
        "answerable_questions": len(answerable),
        "unanswerable_questions": len(refusals),
        "correct_answerable": correct_answerable,
        "correct_unanswerable": correct_refusals,
        "mean_judge_score_answerable": (
            round(correct_answerable / len(answerable), 4) if answerable else 0.0
        ),
        "correct_refusal_rate": (
            round(correct_refusals / len(refusals), 4) if refusals else 0.0
        ),
        "false_refusal_rate": (
            round(false_refusals / len(answerable), 4) if answerable else 0.0
        ),
        "overall_mean_match_score": (
            round((correct_answerable + correct_refusals) / total, 4) if total else 0.0
        ),
        "per_question_type": per_type,
        "retrieval": retrieval_metric_aggregate(item_results),
    }


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class DeliveryPolicy:
    """Explicit native-media/context delivery limits."""

    mode: str = "production_budget"
    max_images: int | None = 5
    max_audio: int | None = 3
    max_video: int | None = 1
    max_media_size_mb: float | None = 20.0
    max_context_chars: int | None = None
    supported_modalities: set[str] = field(
        default_factory=lambda: {"text", "image", "audio", "video"}
    )
    attach_media: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> "DeliveryPolicy":
        config = config or {}
        budget = config.get("delivery_budget") or config.get("delivery") or config
        mode = str(budget.get("mode", config.get("delivery_mode", "production_budget")))
        if mode not in {"all_retrieved_delivered", "production_budget"}:
            raise ValueError(f"Unknown delivery mode: {mode}")
        no_local_caps = mode == "all_retrieved_delivered"
        # None means no local cap; provider limits can still reject a block.
        return cls(
            mode=mode,
            max_images=(None if no_local_caps else _safe_int(
                budget.get("max_images", budget.get("max_images_per_query")), 5
            )),
            max_audio=(None if no_local_caps else _safe_int(
                budget.get("max_audio", budget.get("max_audio_per_query")), 3
            )),
            max_video=(None if no_local_caps else _safe_int(
                budget.get("max_video", budget.get("max_video_per_query")), 1
            )),
            max_media_size_mb=(
                float(budget["max_media_size_mb"])
                if budget.get("max_media_size_mb") is not None
                else 20.0
            ),
            max_context_chars=_safe_int(budget.get("max_context_chars")),
            supported_modalities=set(budget.get("supported_modalities", MEDIA_MODALITIES | {"text"})),
            attach_media=bool(budget.get("attach_media", True)),
        )

    def cap_for(self, modality: str) -> int | None:
        return {"image": self.max_images, "audio": self.max_audio, "video": self.max_video}.get(modality)


def _unit_manifest(unit: Mapping[str, Any], rank: int) -> dict[str, Any]:
    facts = sorted(_fact_keys(unit))
    return {
        "rank": rank,
        "chunk_id": str(unit.get("chunk_id") or unit.get("id") or ""),
        "session_id": str(unit.get("session_id") or unit.get("source_session_id") or ""),
        "timestamp": unit.get("timestamp", unit.get("session_date", "")),
        "modality": str(unit.get("modality") or "text"),
        "fact_keys": facts,
        "score": unit.get("score"),
        "media_path": unit.get("media_path"),
        "content": unit.get("content", unit.get("text", "")) or "",
    }


def pack_evidence(
    ranked_units: Sequence[Mapping[str, Any]],
    policy: DeliveryPolicy | Mapping[str, Any] | None = None,
    *,
    provider_blocks: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pack ranked units and return provider blocks plus a delivery manifest.

    ``provider_blocks`` can be supplied by a backend-specific serializer.  The
    packer still makes all inclusion/drop decisions and never treats a text
    description as successful native-media delivery.
    """
    # Accept Mapping/None via from_config. Also accept duck-typed DeliveryPolicy
    # instances from the dual import paths used by evaluation entrypoints
    # (`protocol` vs `evaluation.protocol`); a failed isinstance check used to
    # fall through to from_config(None) and silently reset caps to defaults.
    if policy is None or isinstance(policy, Mapping):
        policy = DeliveryPolicy.from_config(policy)
    elif not isinstance(policy, DeliveryPolicy):
        if not callable(getattr(policy, "cap_for", None)):
            raise TypeError(
                f"Unsupported delivery policy type: {type(policy)!r}"
            )
    supplied_blocks = list(provider_blocks or [])
    delivered: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    used_chars = 0
    for position, raw_unit in enumerate(ranked_units, start=1):
        unit = _unit_manifest(raw_unit, position)
        modality = unit["modality"]
        reason: str | None = None
        media_path = unit.get("media_path")
        if modality not in policy.supported_modalities:
            reason = "provider_unsupported"
        elif modality in MEDIA_MODALITIES and not policy.attach_media:
            reason = "provider_unsupported"
        elif modality in MEDIA_MODALITIES:
            cap = policy.cap_for(modality)
            already = sum(1 for entry in delivered if entry["modality"] == modality and entry.get("native_attached"))
            if cap is not None and already >= cap:
                reason = "media_count_cap"
            elif not media_path or not os.path.isfile(str(media_path)):
                reason = "missing_file"
            elif policy.max_media_size_mb is not None:
                try:
                    if os.path.getsize(str(media_path)) > policy.max_media_size_mb * 1024 * 1024:
                        reason = "media_size_cap"
                except OSError:
                    reason = "missing_file"
        content = str(unit.get("content") or "")
        if reason is None and policy.max_context_chars is not None:
            if used_chars + len(content) > policy.max_context_chars:
                reason = "context_token_cap"
        if reason is not None:
            unit["drop_reason"] = reason
            unit["native_attached"] = False
            unit["delivery_status"] = "dropped"
            dropped.append(unit)
            counts[reason] += 1
            continue
        unit["native_attached"] = modality in MEDIA_MODALITIES
        unit["delivery_status"] = "attached" if unit["native_attached"] else "text_only"
        delivered.append(unit)
        used_chars += len(content)
        block_index = len(blocks)
        if block_index < len(supplied_blocks):
            blocks.append(dict(supplied_blocks[block_index]))
        else:
            block = {"type": "text", "text": content}
            if unit["native_attached"]:
                block["modality"] = modality
                block["media_path"] = media_path
            blocks.append(block)
    return {
        "mode": policy.mode,
        "provider_blocks": blocks,
        "delivered": delivered,
        "dropped": dropped,
        "failed": failed,
        "drop_reasons": dict(counts),
        "effective_context_chars": used_chars,
        "attachment_counts": {
            modality: sum(1 for unit in delivered if unit["modality"] == modality and unit.get("native_attached"))
            for modality in sorted(MEDIA_MODALITIES)
        },
    }


def _encode_image_jpeg_b64(path: str, *, max_bytes: int | None = None) -> tuple[str, str]:
    """Re-encode an image as JPEG base64 to keep provider payloads small."""
    from io import BytesIO

    from PIL import Image

    image = Image.open(path).convert("RGB")
    quality = 85
    raw = b""
    for quality in (85, 70, 55):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        raw = buffer.getvalue()
        if max_bytes is None or len(raw) <= max_bytes:
            break
    if max_bytes is not None and len(raw) > max_bytes:
        raise ValueError(
            f"JPEG-compressed image still exceeds {max_bytes} bytes: {path}"
        )
    return base64.b64encode(raw).decode("ascii"), "image/jpeg"


def materialize_openai_media_blocks(
    delivery: Mapping[str, Any],
    *,
    api_format: str | None = None,
    max_media_size_mb: float | None = 28.0,
) -> dict[str, Any]:
    """Serialize attached media and update the manifest on read failures."""
    result = {
        key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, Mapping) else value)
        for key, value in delivery.items()
    }
    kept: list[dict[str, Any]] = []
    failed = list(result.get("failed", []))
    blocks: list[dict[str, Any]] = []
    reasons = Counter(result.get("drop_reasons", {}))
    max_bytes = (
        int(max_media_size_mb * 1024 * 1024)
        if max_media_size_mb is not None
        else None
    )
    for raw_unit in result.get("delivered", []):
        unit = dict(raw_unit)
        if not unit.get("native_attached"):
            kept.append(unit)
            continue
        path = str(unit.get("media_path") or "")
        modality = str(unit.get("modality") or "")
        try:
            if modality == "image":
                encoded, mime = _encode_image_jpeg_b64(path, max_bytes=max_bytes)
            else:
                with open(path, "rb") as handle:
                    raw = handle.read()
                if max_bytes is not None and len(raw) > max_bytes:
                    raise ValueError(
                        f"Media file exceeds {max_bytes} bytes after read: {path}"
                    )
                encoded = base64.b64encode(raw).decode("ascii")
                mime = mimetypes.guess_type(path)[0] or {
                    "audio": "audio/mpeg",
                    "video": "video/mp4",
                }.get(modality, "application/octet-stream")
            data_url = f"data:{mime};base64,{encoded}"
            if modality == "image":
                block = {"type": "image_url", "image_url": {"url": data_url}}
            elif modality == "audio":
                extension = Path(path).suffix.lower().lstrip(".") or "mp3"
                block = {
                    "type": "input_audio",
                    "input_audio": {"data": encoded, "format": extension},
                }
            elif modality == "video":
                block_type = "image_url" if api_format == "gemini" else "video_url"
                block = {block_type: {"url": data_url}, "type": block_type}
            else:
                raise ValueError(f"Unsupported media modality: {modality}")
        except Exception as error:
            unit["native_attached"] = False
            unit["delivery_status"] = "failed"
            unit["drop_reason"] = "serialization_error"
            unit["error_class"] = type(error).__name__
            unit["error_message"] = str(error)
            failed.append(unit)
            reasons["serialization_error"] += 1
            continue
        blocks.append(block)
        kept.append(unit)
    result["delivered"] = kept
    result["failed"] = failed
    result["drop_reasons"] = dict(reasons)
    result["provider_blocks"] = blocks
    result["attachment_counts"] = {
        modality: sum(
            1 for unit in kept
            if unit.get("modality") == modality and unit.get("native_attached")
        )
        for modality in sorted(MEDIA_MODALITIES)
    }
    return result


def delivery_metrics(
    delivered: Sequence[Mapping[str, Any]],
    evidence_plan: Sequence[Mapping[str, Any]],
    *,
    required_session_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute the same fact/chain metrics over delivered units."""
    required, _ = _required_fact_info(evidence_plan)
    if not required:
        return {
            "status": "not_applicable",
            "required_fact_coverage": "not_applicable",
            "complete_chain_covered": "not_applicable",
            "delivered_fact_keys": [],
        }
    present: set[str] = set()
    native_present: set[str] = set()
    sessions: set[str] = set()
    for unit in delivered:
        facts = _fact_keys(unit)
        present.update(facts)
        if unit.get("native_attached"):
            native_present.update(facts)
        if _session_id(unit):
            sessions.add(_session_id(unit) or "")
    # Native facts are required for a media carrier; text units can satisfy
    # text facts as usual.  The caller can inspect both sets for diagnosis.
    coverage = len(present & required) / len(required)
    return {
        "status": "ok",
        "required_fact_coverage": round(coverage, 4),
        "complete_chain_covered": required.issubset(present),
        "delivered_fact_keys": sorted(present & required),
        "native_delivered_fact_keys": sorted(native_present & required),
        "required_fact_keys": sorted(required),
        "required_session_ids": sorted({str(x) for x in (required_session_ids or ()) if x}),
        "delivered_session_ids": sorted(sessions),
    }


def make_item_trace(
    *,
    question_id: str,
    query: Mapping[str, Any] | None = None,
    ranked: Sequence[Mapping[str, Any]] = (),
    evidence_plan: Sequence[Mapping[str, Any]] = (),
    delivery: Mapping[str, Any] | None = None,
    answer: Mapping[str, Any] | None = None,
    react: Mapping[str, Any] | None = None,
    required_session_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build the stable four-stage per-item trace schema."""
    ranked_manifest = [_unit_manifest(unit, rank) for rank, unit in enumerate(ranked, start=1)]
    retrieval = compute_retrieval_recall(
        ranked_manifest,
        evidence_plan,
        required_session_ids=required_session_ids,
    )
    packed = dict(delivery or pack_evidence(ranked_manifest, DeliveryPolicy(mode="all_retrieved_delivered")))
    delivered = list(packed.get("delivered", []))
    delivered_stats = delivery_metrics(
        delivered,
        evidence_plan,
        required_session_ids=required_session_ids,
    )
    retrieval["fact_recall_at_k"] = retrieval.get("fact_recall_at_k", {})
    return {
        "question_id": question_id,
        "query": {
            "original": (query or {}).get("original", ""),
            "retrieval_query": (query or {}).get("retrieval_query", ""),
            "rewrite_model": (query or {}).get("rewrite_model"),
        },
        "retrieval": {
            "ranked": ranked_manifest,
            "required_fact_keys": retrieval.get("required_fact_keys", []),
            "fact_recall_at_k": retrieval.get("fact_recall_at_k", {}),
            "complete_chain_at_k": retrieval.get("complete_chain_at_k", {}),
            "required_session_recall_at_k": retrieval.get("required_session_recall_at_k", {}),
            "status": retrieval.get("status"),
        },
        "delivery": {
            "delivered": delivered,
            "dropped": list(packed.get("dropped", [])),
            "failed": list(packed.get("failed", [])),
            "drop_reasons": dict(packed.get("drop_reasons", {})),
            "required_fact_coverage": delivered_stats.get("required_fact_coverage", "not_applicable"),
            "complete_chain_covered": delivered_stats.get("complete_chain_covered", "not_applicable"),
            "attachment_counts": packed.get("attachment_counts", {}),
            "effective_context_chars": packed.get("effective_context_chars", 0),
        },
        "answer": {
            "predicted": (answer or {}).get("predicted", ""),
            "api_status": (answer or {}).get("api_status", "empty"),
            "forced": bool((answer or {}).get("forced", False)),
            **{k: v for k, v in (answer or {}).items() if k not in {"predicted", "api_status", "forced"}},
        },
        "react": react,
    }


def _secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in ("api_key", "token", "secret", "password", "credential"))


def redact_config(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe config with secret values removed."""
    if isinstance(value, Mapping):
        return {
            str(k): (
                "<redacted>"
                if _secret_key(str(k))
                else redact_config(v, key=str(k))
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_config(v, key=key) for v in value]
    if isinstance(value, tuple):
        return [redact_config(v, key=key) for v in value]
    return _json_safe(value)


def git_commit(cwd: str | os.PathLike[str] | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def write_run_manifest(
    output_dir: str | os.PathLike[str],
    config: Mapping[str, Any] | None = None,
    *,
    canonical_manifest: str | os.PathLike[str] | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """Write a redacted, self-describing ``run_manifest.json``."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    redacted = redact_config(dict(config or {}))
    manifest_path = str(canonical_manifest or DEFAULT_QID_MANIFEST)
    canonical_hash = "unknown"
    canonical_qids = None
    if os.path.exists(manifest_path):
        try:
            qid_manifest = load_qid_manifest(manifest_path)
            canonical_hash = qid_manifest["question_id_sha256"]
            canonical_qids = manifest_path
        except Exception:
            canonical_hash = sha256_file(manifest_path)
            canonical_qids = manifest_path
    requested_count = fields.pop("num_requested_items", 923)
    evaluated_count = fields.pop("evaluated_item_count", requested_count)
    defaults = {
        "run_id": f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:8]}",
        "git_commit": git_commit(Path(__file__).resolve().parent.parent),
        "canonical_qid_manifest": canonical_qids or manifest_path,
        "canonical_qid_sha256": canonical_hash,
        "num_requested_items": requested_count,
        "memory_snapshot": "final",
        "memory_scope": "persona",
        "execution_track": "fixed_reader",
        "memory_organization": "flat_index",
        "store_source_item_count": fields.pop("store_source_item_count", 1004),
        "evaluated_item_count": evaluated_count,
        "representation": "unknown",
        "index_manifest": None,
        "retriever": "unknown",
        "native_system_reader": None,
        "chunk_unit": "session",
        "query_policy": "original",
        "answer_reasoner": None,
        "judge": None,
        "top_k": None,
        "delivery_budget": {},
        "config_sha256": sha256_json(redacted),
        "config": redacted,
    }
    defaults.update(fields)
    with (destination / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(defaults, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return defaults


def write_details_jsonl(
    output_dir: str | os.PathLike[str],
    results: Sequence[Mapping[str, Any]],
    *,
    filename: str = "details.jsonl",
) -> str:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / filename
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(dict(result), ensure_ascii=False) + "\n")
    return str(path)


def normalized_message_payload(session: Mapping[str, Any]) -> str:
    messages = session.get("messages") or []
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": message.get("role", ""),
                "content": message.get("content", ""),
                "multimodal_content": message.get("multimodal_content"),
            }
        )
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def audit_persona_session_collisions(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit repeated session IDs and deterministic content identities."""
    by_session_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    occurrence_map: dict[str, str] = {}
    fact_by_session: dict[str, set[str]] = defaultdict(set)
    for record in records:
        item = record.get("item") or {}
        persona = str(record.get("persona_uuid") or "")
        required_keys = {
            str(evidence.get("fact_key"))
            for evidence in item.get("evidence_plan") or []
            if evidence.get("required_for_answer") and evidence.get("fact_key")
        }
        for session in item.get("haystack_sessions") or []:
            sid = str(session.get("session_id") or "")
            if not sid:
                continue
            timestamp = str(session.get("timestamp", session.get("session_date", "")))
            payload = normalized_message_payload(session)
            session_identity = sha256_bytes(payload.encode("utf-8"))
            occurrence_identity = f"{session_identity}:{timestamp}"
            existing = occurrence_map.get(f"{persona}:{occurrence_identity}")
            if existing is not None and existing != payload:
                raise ValueError(
                    f"Occurrence identity collision for persona={persona}, session={sid}, timestamp={timestamp}"
                )
            occurrence_map[f"{persona}:{occurrence_identity}"] = payload
            facts: set[str] = set()
            for message in session.get("messages") or []:
                facts.update(_fact_keys(message) & required_keys)
            fact_by_session[f"{persona}:{sid}"].update(facts)
            by_session_id[f"{persona}:{sid}"].append(
                {
                    "session_id": sid,
                    "timestamp": timestamp,
                    "session_identity": session_identity,
                    "occurrence_identity": occurrence_identity,
                    "payload_sha256": sha256_bytes(payload.encode("utf-8")),
                }
            )
    repeated = {key: values for key, values in by_session_id.items() if len(values) > 1}
    conflicting = {
        key: values
        for key, values in repeated.items()
        if len({value["payload_sha256"] for value in values}) > 1
    }
    affected = {
        key: sorted(fact_by_session.get(key, set()))
        for key in repeated
        if fact_by_session.get(key)
    }
    return {
        "num_session_occurrences": sum(len(v) for v in by_session_id.values()),
        "num_unique_session_ids": len(by_session_id),
        "num_repeated_session_ids": len(repeated),
        "num_conflicting_repeated_ids": len(conflicting),
        "repeated_session_ids": repeated,
        "conflicting_session_ids": conflicting,
        "repeated_ids_with_required_facts": affected,
        "identity_policy": "session_identity=sha256(normalized message payload); occurrence_identity=session_identity+timestamp",
    }
