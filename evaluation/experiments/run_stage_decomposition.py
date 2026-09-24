#!/usr/bin/env python3
"""Run the paired WRITE--RETRIEVE--DELIVER--READ decomposition.

This is the single entry point for Experiment 2 in
``docs/EVALUATION_IMPLEMENTATION_PLAN.md``. All conditions use the same
answer reader and prompt; only the evidence builder and delivery policy vary.
The primary population is the canonical ordered set of 784 answerable items.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from loguru import logger
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
EVALUATION_ROOT = HERE.parent
REPO_ROOT = EVALUATION_ROOT.parent
for import_root in (str(EVALUATION_ROOT), str(HERE)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from _common import (  # noqa: E402
    aggregate_results,
    encode_omni_queries,
    load_config,
    load_faiss_index,
    match_answer,
    search_faiss_index_filtered,
)
from answer_harness import (  # noqa: E402
    ANSWER_PROMPT_VERSION,
    CommonAnswerHarness,
    ObservableChatRunner,
)
from protocol import (  # noqa: E402
    DeliveryPolicy,
    canonical_qid_sha256,
    compute_retrieval_recall,
    load_canonical_records,
    make_item_trace,
    materialize_openai_media_blocks,
    pack_evidence,
    records_by_persona,
    sha256_file,
    sha256_json,
    validate_result_qids,
    write_details_jsonl,
    write_run_manifest,
)


CONDITIONS = (
    "gold_symbolic_facts",
    "gold_raw_evidence",
    "gold_generic_text",
    "gold_structured_text",
    "retrieved_raw_all_delivered",
    "retrieved_raw_budgeted",
    "end_to_end",
)
GOLD_CONDITIONS = {
    "gold_symbolic_facts",
    "gold_raw_evidence",
    "gold_generic_text",
    "gold_structured_text",
}
RETRIEVED_CONDITIONS = {
    "retrieved_raw_all_delivered",
    "retrieved_raw_budgeted",
    "end_to_end",
}
STRUCTURED_WRITER_VERSION = "query-blind-modality-schema-v1"
STRUCTURED_SCHEMAS = {
    "image": (
        "Describe the image without knowing any downstream question. Return JSON "
        "with keys scene, objects, visible_text, spatial_relations, attributes, "
        "counts, and uncertainty. Preserve fine detail and relative positions."
    ),
    "video": (
        "Describe the video without knowing any downstream question. Return JSON "
        "with keys setting, entities, ordered_events, state_changes, counts, "
        "visible_text, and uncertainty. Preserve temporal order and repetitions."
    ),
    "audio": (
        "Describe the audio without knowing any downstream question. Return JSON "
        "with keys transcript, speakers, delivery, emotion_or_prosody, acoustic_events, "
        "timing, and uncertainty. Keep wording distinct from vocal delivery."
    ),
}


def _read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: str | os.PathLike[str], value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def _fact_keys(unit: Mapping[str, Any]) -> list[str]:
    values = unit.get("fact_keys")
    if isinstance(values, str):
        values = [values]
    if not values:
        value = unit.get("fact_key")
        values = value if isinstance(value, list) else [value] if value else []
    return sorted({str(value) for value in values if value})


def _required_entries(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    entries = [
        dict(entry)
        for entry in item.get("evidence_plan", [])
        if entry.get("required_for_answer") and entry.get("available_at_query", True)
    ]
    return sorted(entries, key=lambda entry: int(entry.get("hop_index") or 0))


def _carrier_map(item: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    required = {entry["fact_key"] for entry in _required_entries(item)}
    carriers: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for session in item.get("haystack_sessions", item.get("sessions", [])):
        for message in session.get("messages", []):
            mm = message.get("multimodal_content") or {}
            fact_key = message.get("fact_key") or mm.get("fact_key")
            if fact_key not in required:
                continue
            if fact_key in carriers:
                raise ValueError(
                    f"Required fact {fact_key} appears in multiple carrier messages"
                )
            carriers[str(fact_key)] = (dict(session), dict(message))
    missing = sorted(required - set(carriers))
    if missing:
        raise ValueError(f"Missing required carrier messages: {missing}")
    return carriers


def _unit(
    *,
    chunk_id: str,
    session: Mapping[str, Any],
    modality: str,
    content: str,
    fact_key: str,
    media_path: str | None = None,
    score: float | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "session_id": str(session.get("session_id") or ""),
        "timestamp": session.get("timestamp", session.get("session_date", "")),
        "modality": modality,
        "content": content,
        "fact_key": fact_key,
        "fact_keys": [fact_key],
        "media_path": media_path,
        "score": score,
    }


def build_gold_symbolic_facts(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build answer-blind symbolic statements with no internal role labels."""
    carriers = _carrier_map(item)
    units: list[dict[str, Any]] = []
    for index, entry in enumerate(_required_entries(item), start=1):
        fact_key = str(entry["fact_key"])
        session, _ = carriers[fact_key]
        statement = str(entry.get("fact_statement") or "").strip()
        value = str(entry.get("value") or "").strip()
        content = f"Statement: {statement}\nObserved value: {value}"
        units.append(_unit(
            chunk_id=f"symbolic:{item['question_id']}:{index}",
            session=session,
            modality="text",
            content=content,
            fact_key=fact_key,
        ))
    return units


def build_gold_raw_evidence(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    carriers = _carrier_map(item)
    units: list[dict[str, Any]] = []
    for index, entry in enumerate(_required_entries(item), start=1):
        fact_key = str(entry["fact_key"])
        session, message = carriers[fact_key]
        mm = message.get("multimodal_content") or {}
        media_path = str(mm.get("media_path") or "") or None
        modality = str(mm.get("modality") or mm.get("type") or "text")
        content = str(
            mm.get("natural_utterance") or message.get("content") or ""
        ).strip()
        units.append(_unit(
            chunk_id=f"gold_raw:{item['question_id']}:{index}",
            session=session,
            modality=modality,
            content=content,
            fact_key=fact_key,
            media_path=media_path,
        ))
    return units


class TextualizedCorpus:
    """Lazy per-persona access to existing query-agnostic textualizations."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self._cache: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _load_persona(self, source_filename: str) -> dict[str, dict[str, Any]]:
        tag = source_filename.removeprefix("stage2_with_media_")
        path = self.root / f"stage_2_text_based_{tag}"
        if not path.is_file():
            raise FileNotFoundError(f"Textualized persona file not found: {path}")
        payload = _read_json(path)
        return {str(item["question_id"]): item for item in payload.get("items", [])}

    def item(self, record: Mapping[str, Any]) -> dict[str, Any]:
        source = str(record["source_filename"])
        with self._lock:
            if source not in self._cache:
                self._cache[source] = self._load_persona(source)
            item = self._cache[source].get(str(record["question_id"]))
        if item is None:
            raise KeyError(f"Missing textualized item {record['question_id']} in {source}")
        return item


def build_gold_generic_text(
    record: Mapping[str, Any], corpus: TextualizedCorpus
) -> list[dict[str, Any]]:
    item = corpus.item(record)
    carriers = _carrier_map(item)
    units: list[dict[str, Any]] = []
    for index, entry in enumerate(_required_entries(record["item"]), start=1):
        fact_key = str(entry["fact_key"])
        session, message = carriers[fact_key]
        mm = message.get("multimodal_content") or {}
        text = str(mm.get("textualized") or message.get("content") or "").strip()
        if not text:
            raise ValueError(f"Generic textualization is empty for {record['question_id']}:{fact_key}")
        units.append(_unit(
            chunk_id=f"generic:{record['question_id']}:{index}",
            session=session,
            modality="text",
            content=text,
            fact_key=fact_key,
        ))
    return units


class StructuredWriter:
    """Question-blind modality writer with content-addressed persistent cache."""

    def __init__(self, config: Mapping[str, Any], cache_path: str | os.PathLike[str]) -> None:
        self.config = dict(config)
        self.cache_path = Path(cache_path)
        self.chat = ObservableChatRunner(self.config)
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        if self.cache_path.is_file():
            payload = _read_json(self.cache_path)
            self._entries = {
                str(key): dict(value)
                for key, value in (payload.get("entries") or {}).items()
            }

    @staticmethod
    def cache_key(media_path: str, modality: str, writer_config: Mapping[str, Any]) -> str:
        return sha256_json({
            "media_sha256": sha256_file(media_path),
            "modality": modality,
            "schema": STRUCTURED_SCHEMAS[modality],
            "writer_version": STRUCTURED_WRITER_VERSION,
            "model": writer_config.get("llm_model"),
        })

    def _save_locked(self) -> None:
        _write_json_atomic(self.cache_path, {
            "manifest_version": 1,
            "writer_version": STRUCTURED_WRITER_VERSION,
            "writer_model": self.config.get("llm_model"),
            "num_entries": len(self._entries),
            "entries": dict(sorted(self._entries.items())),
        })

    def describe(self, media_path: str, modality: str) -> tuple[str, dict[str, Any]]:
        if modality not in STRUCTURED_SCHEMAS:
            raise ValueError(f"No structured schema for modality: {modality}")
        key = self.cache_key(media_path, modality, self.config)
        with self._lock:
            cached = self._entries.get(key)
        if cached and cached.get("api_status") == "ok":
            return str(cached["description"]), dict(cached)

        delivery = pack_evidence(
            [{
                "chunk_id": key,
                "session_id": "",
                "timestamp": "",
                "modality": modality,
                "content": "",
                "media_path": media_path,
            }],
            DeliveryPolicy(mode="all_retrieved_delivered", max_media_size_mb=None),
        )
        delivery = materialize_openai_media_blocks(
            delivery, api_format=self.config.get("api_format")
        )
        blocks = delivery.get("provider_blocks", [])
        if not blocks:
            raise ValueError(f"Could not serialize structured-writer media: {media_path}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": STRUCTURED_SCHEMAS[modality]},
                *blocks,
            ],
        }]
        result = self.chat.complete(
            messages, max_tokens=int(self.config.get("max_tokens", 2048))
        )
        entry = {
            "cache_key": key,
            "media_path": media_path,
            "media_sha256": sha256_file(media_path),
            "modality": modality,
            "schema": STRUCTURED_SCHEMAS[modality],
            "description": result.text,
            **result.to_dict(),
        }
        with self._lock:
            self._entries[key] = entry
            self._save_locked()
        if result.api_status != "ok":
            raise RuntimeError(result.error_message or "Structured writer returned no text")
        return result.text, entry

    def inspect(self, media_path: str, modality: str) -> tuple[str, dict[str, Any]]:
        """Validate a writer input and inspect its cache without an API call."""
        if modality not in STRUCTURED_SCHEMAS:
            raise ValueError(f"No structured schema for modality: {modality}")
        if not os.path.isfile(media_path):
            raise FileNotFoundError(media_path)
        key = self.cache_key(media_path, modality, self.config)
        with self._lock:
            cached = dict(self._entries.get(key) or {})
        if cached.get("api_status") == "ok":
            return str(cached.get("description") or ""), cached
        return "[STRUCTURED DESCRIPTION PENDING]", {
            "cache_key": key,
            "media_path": media_path,
            "media_sha256": sha256_file(media_path),
            "modality": modality,
            "schema": STRUCTURED_SCHEMAS[modality],
            "api_status": "pending",
            "cached": False,
        }


def build_gold_structured_text(
    item: Mapping[str, Any], writer: StructuredWriter, *, generate: bool = True
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    carriers = _carrier_map(item)
    units: list[dict[str, Any]] = []
    writer_records: list[dict[str, Any]] = []
    for index, entry in enumerate(_required_entries(item), start=1):
        fact_key = str(entry["fact_key"])
        session, message = carriers[fact_key]
        mm = message.get("multimodal_content") or {}
        modality = str(mm.get("modality") or mm.get("type") or "text")
        if modality in STRUCTURED_SCHEMAS:
            media_path = str(mm.get("media_path") or "")
            if not media_path:
                raise ValueError(f"Missing media path for {item['question_id']}:{fact_key}")
            if generate:
                text, writer_record = writer.describe(media_path, modality)
            else:
                text, writer_record = writer.inspect(media_path, modality)
            writer_records.append(writer_record)
        else:
            text = str(message.get("content") or "").strip()
        units.append(_unit(
            chunk_id=f"structured:{item['question_id']}:{index}",
            session=session,
            modality="text",
            content=text,
            fact_key=fact_key,
        ))
    return units, writer_records


class RetrievalManifest:
    """Frozen diagnostic and production-query rankings for retrieved conditions."""

    def __init__(self, path: str | os.PathLike[str], config: Mapping[str, Any]) -> None:
        self.path = Path(path)
        self.config = dict(config)
        self._lock = threading.Lock()
        self._payload: dict[str, Any] = {
            "manifest_version": 2,
            "query_policies": ["original", "fixed_rewrite"],
            "top_k": int(config.get("top_k", 50)),
            "index_dir": str(config["index_dir"]),
            "index_manifest_sha256": sha256_file(
                Path(config["index_dir"]) / "manifest.json"
            ),
            "embedding_model": config.get("embedding", {}).get("model_name"),
            "production_query_manifest": str(config["production_query_manifest"]),
            "production_query_manifest_sha256": sha256_file(
                config["production_query_manifest"]
            ),
            "items": {},
        }
        if self.path.is_file():
            existing = _read_json(self.path)
            for key in (
                "manifest_version", "query_policies", "top_k", "index_dir",
                "index_manifest_sha256", "embedding_model",
                "production_query_manifest",
                "production_query_manifest_sha256",
            ):
                if existing.get(key) != self._payload.get(key):
                    raise ValueError(
                        f"Retrieval manifest setting mismatch for {key}: "
                        f"{existing.get(key)!r} != {self._payload.get(key)!r}"
                    )
            self._payload = existing

    @property
    def entries(self) -> dict[str, Any]:
        return self._payload["items"]

    def validate_qids(self, expected_qids: Sequence[str]) -> None:
        actual = list(self.entries)
        if actual != list(expected_qids):
            raise ValueError(
                f"Retrieval manifest QIDs differ from requested ordered set: "
                f"expected={len(expected_qids)}, actual={len(actual)}"
            )
        top_k = int(self._payload["top_k"])
        for question_id in expected_qids:
            entry = self.entries[question_id]
            for ranking in ("diagnostic", "production"):
                query = str(
                    entry.get(f"{ranking}_retrieval_query") or ""
                ).strip()
                units = entry.get(f"{ranking}_ranked")
                if not query or not isinstance(units, list) or len(units) != top_k:
                    raise ValueError(
                        f"Invalid frozen {ranking} ranking for {question_id}: "
                        f"query={bool(query)}, units="
                        f"{len(units) if isinstance(units, list) else 'invalid'}, "
                        f"expected={top_k}"
                    )

    def save(self) -> None:
        with self._lock:
            self._payload["num_items"] = len(self.entries)
            self._payload["question_id_sha256"] = canonical_qid_sha256(
                self.entries
            )
            _write_json_atomic(self.path, self._payload)

    def build(self, records: Sequence[Mapping[str, Any]], *, force: bool = False) -> None:
        expected_qids = [str(record["question_id"]) for record in records]
        if self.entries and not force and list(self.entries) == expected_qids:
            self.validate_qids(expected_qids)
            return
        if force:
            self._payload["items"] = {}
        query_payload = _read_json(self.config["production_query_manifest"])
        production_queries = query_payload.get("queries", query_payload)
        if not isinstance(production_queries, Mapping):
            raise ValueError("Production query manifest must contain a queries mapping")
        missing_queries = [
            qid for qid in expected_qids if qid not in production_queries
        ]
        if missing_queries:
            raise ValueError(
                "Production query manifest is incomplete; missing fixed rewrites "
                f"for {len(missing_queries)} QIDs, including {missing_queries[:5]}"
            )
        grouped = records_by_persona(records)
        index_dir = Path(self.config["index_dir"])
        top_k = int(self.config.get("top_k", 50))
        embedding = dict(self.config.get("embedding") or {})
        for persona_uuid, persona_records in tqdm(
            grouped.items(), desc="Retrieval manifest", unit="persona"
        ):
            persona_dir = index_dir / persona_uuid
            index, metadata = load_faiss_index(
                str(persona_dir / "index.faiss"),
                str(persona_dir / "metadata.json"),
            )
            missing_records = [
                record for record in persona_records
                if str(record["question_id"]) not in self.entries
            ]
            if not missing_records:
                continue
            diagnostic_queries = [
                str(record["item"].get("question") or "")
                for record in missing_records
            ]
            production_texts: list[str] = []
            for record, original in zip(missing_records, diagnostic_queries):
                qid = str(record["question_id"])
                query_entry = production_queries[qid]
                if not isinstance(query_entry, Mapping):
                    raise ValueError(f"Invalid production query entry for {qid}")
                if query_entry.get("original") not in {None, original}:
                    raise ValueError(
                        f"Production query manifest question mismatch for {qid}"
                    )
                rewritten = str(
                    query_entry.get("retrieval_query") or ""
                ).strip()
                if not rewritten:
                    raise ValueError(f"Empty production retrieval query for {qid}")
                production_texts.append(rewritten)
            diagnostic_vectors = encode_omni_queries(
                diagnostic_queries, embedding
            ).astype("float32")
            production_vectors = encode_omni_queries(
                production_texts, embedding
            ).astype("float32")

            def ranked_units(vector: Any) -> list[dict[str, Any]]:
                units = search_faiss_index_filtered(
                    index, vector, metadata, k=top_k, candidate_k=top_k
                )
                return [
                    {
                        "rank": rank,
                        "chunk_id": raw.get("chunk_id", ""),
                        "session_id": raw.get("session_id", ""),
                        "timestamp": raw.get("timestamp", ""),
                        "modality": raw.get("modality", "text"),
                        "content": raw.get("content", ""),
                        "media_path": raw.get("media_path"),
                        "fact_key": raw.get("fact_key"),
                        "fact_keys": _fact_keys(raw),
                        "score": raw.get("score"),
                        "pre_filter_rank": raw.get("pre_filter_rank"),
                        "eligible_rank": raw.get("eligible_rank"),
                    }
                    for rank, raw in enumerate(units, start=1)
                ]

            for record, original, production_query, diagnostic_vector, production_vector in zip(
                missing_records,
                diagnostic_queries,
                production_texts,
                diagnostic_vectors,
                production_vectors,
            ):
                question_id = str(record["question_id"])
                self.entries[question_id] = {
                    "question_id": question_id,
                    "persona_uuid": persona_uuid,
                    "original_query": original,
                    "diagnostic_retrieval_query": original,
                    "production_retrieval_query": production_query,
                    "production_rewrite_model": (
                        production_queries[question_id].get("rewrite_model")
                    ),
                    "diagnostic_ranked": ranked_units(diagnostic_vector),
                    "production_ranked": ranked_units(production_vector),
                }
            self.save()
        self._payload["items"] = {
            qid: self.entries[qid] for qid in expected_qids if qid in self.entries
        }
        self.save()
        self.validate_qids([str(record["question_id"]) for record in records])


def retrieval_units(
    record: Mapping[str, Any],
    manifest: RetrievalManifest,
    *,
    ranking: str = "diagnostic",
) -> list[dict[str, Any]]:
    entry = manifest.entries.get(str(record["question_id"]))
    if entry is None:
        raise KeyError(f"Missing retrieval manifest entry: {record['question_id']}")
    if entry.get("original_query") != record["item"].get("question"):
        raise ValueError(f"Retrieval query mismatch: {record['question_id']}")
    if ranking not in {"diagnostic", "production"}:
        raise ValueError(f"Unknown retrieval ranking: {ranking}")
    key = f"{ranking}_ranked"
    units = [dict(unit) for unit in entry.get(key, [])]
    if not units:
        raise ValueError(
            f"Frozen {ranking} ranking is empty: {record['question_id']}"
        )
    return units


def retrieval_query_details(
    record: Mapping[str, Any],
    manifest: RetrievalManifest,
    *,
    ranking: str,
) -> dict[str, Any]:
    entry = manifest.entries.get(str(record["question_id"])) or {}
    if ranking == "production":
        return {
            "original": entry.get("original_query", ""),
            "retrieval_query": entry.get("production_retrieval_query", ""),
            "rewrite_model": entry.get("production_rewrite_model"),
        }
    return {
        "original": entry.get("original_query", ""),
        "retrieval_query": entry.get("diagnostic_retrieval_query", ""),
        "rewrite_model": None,
    }


def _optional_cap(raw: Mapping[str, Any], key: str) -> int | None:
    """Return an int cap when present; None means uncapped."""
    if key not in raw or raw[key] is None:
        return None
    return int(raw[key])


def _delivery_policy(condition: str, config: Mapping[str, Any]) -> DeliveryPolicy:
    if condition in {
        "gold_symbolic_facts", "gold_generic_text", "gold_structured_text"
    }:
        return DeliveryPolicy(
            mode="all_retrieved_delivered",
            supported_modalities={"text"},
            attach_media=False,
            max_media_size_mb=None,
        )
    if condition in {"gold_raw_evidence", "retrieved_raw_all_delivered"}:
        # Keep mode=all_retrieved_delivered (no production_top_k / production_delivery
        # budget). Optional provider_delivery_caps enforce hard API limits (e.g.
        # Gemini ≤10 videos/request) without collapsing into production_budget.
        caps = dict(config.get("provider_delivery_caps") or {})
        max_media_size_mb = (
            float(caps["max_media_size_mb"])
            if caps.get("max_media_size_mb") is not None
            else None
        )
        return DeliveryPolicy(
            mode="all_retrieved_delivered",
            max_images=_optional_cap(caps, "max_images"),
            max_audio=_optional_cap(caps, "max_audio"),
            max_video=_optional_cap(caps, "max_video"),
            max_media_size_mb=max_media_size_mb,
        )
    budget = dict(config.get("production_delivery") or {})
    budget["mode"] = "production_budget"
    return DeliveryPolicy.from_config(budget)


def _pack(condition: str, units: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    delivered_units = list(units)
    if condition in {"retrieved_raw_budgeted", "end_to_end"}:
        delivered_units = delivered_units[
            : int(config.get("production_top_k", 20))
        ]
    delivery = pack_evidence(
        delivered_units, _delivery_policy(condition, config)
    )
    if any(unit.get("native_attached") for unit in delivery.get("delivered", [])):
        delivery = materialize_openai_media_blocks(
            delivery, api_format=config.get("reader", {}).get("api_format")
        )
    else:
        delivery["provider_blocks"] = []
    return delivery


def _terminal_record(
    record: Mapping[str, Any], condition: str, error: Exception | str
) -> dict[str, Any]:
    item = record["item"]
    message = str(error)
    trace = make_item_trace(
        question_id=str(record["question_id"]),
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
    return {
        "question_id": record["question_id"],
        "persona_uuid": record["persona_uuid"],
        "question_type": item.get("question_type"),
        "answerability": item.get("answerability"),
        "question": item.get("question"),
        "expected_answer": item.get("answer"),
        "answer_aliases": item.get("answer_aliases", []),
        "condition": condition,
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


class StageDecompositionRunner:
    def __init__(
        self,
        config: Mapping[str, Any],
        condition: str,
        records: Sequence[Mapping[str, Any]],
        output_dir: str | os.PathLike[str],
        retrieval: RetrievalManifest | None = None,
    ) -> None:
        self.config = dict(config)
        self.condition = condition
        self.records = list(records)
        self.output_dir = Path(output_dir)
        self.harness = CommonAnswerHarness(self.config["reader"])
        self.generic_corpus = TextualizedCorpus(self.config["generic_text_dir"])
        self.writer = StructuredWriter(
            self.config["structured_writer"], self.config["structured_cache"]
        )
        self.retrieval = retrieval
        if self.condition in RETRIEVED_CONDITIONS and self.retrieval is None:
            retrieval_config = {
                "index_dir": self.config["native_index_dir"],
                "top_k": self.config.get("retrieval_top_k", 50),
                "embedding": self.config.get("embedding", {}),
            }
            self.retrieval = RetrievalManifest(
                self.config["retrieval_manifest"], retrieval_config
            )

    def build_units(
        self, record: Mapping[str, Any], *, generate_structured: bool = True
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        item = record["item"]
        auxiliary: dict[str, Any] = {}
        if self.condition == "gold_symbolic_facts":
            units = build_gold_symbolic_facts(item)
        elif self.condition == "gold_raw_evidence":
            units = build_gold_raw_evidence(item)
        elif self.condition == "gold_generic_text":
            units = build_gold_generic_text(record, self.generic_corpus)
        elif self.condition == "gold_structured_text":
            units, writer_records = build_gold_structured_text(
                item, self.writer, generate=generate_structured
            )
            auxiliary["structured_writer"] = writer_records
        elif self.condition in RETRIEVED_CONDITIONS:
            if self.retrieval is None:
                raise RuntimeError(
                    f"Retrieval state is unavailable for {self.condition}"
                )
            ranking = "production" if self.condition == "end_to_end" else "diagnostic"
            units = retrieval_units(record, self.retrieval, ranking=ranking)
        else:
            raise ValueError(f"Unknown condition: {self.condition}")
        return units, auxiliary

    def evaluate_one(self, record: Mapping[str, Any]) -> dict[str, Any]:
        item = record["item"]
        started = time.time()
        try:
            units, auxiliary = self.build_units(record)
            delivery = _pack(self.condition, units, self.config)
            answer = self.harness.answer(
                str(item.get("question") or ""),
                delivery.get("delivered", []),
                delivery.get("provider_blocks", []),
                question_type=item.get("question_type"),
            )
            if answer.api_status == "ok":
                evaluation = match_answer(
                    answer.text,
                    str(item.get("answer") or ""),
                    item.get("answer_aliases", []),
                    "answerable",
                )
            else:
                evaluation = {
                    "exact_match": False,
                    "alias_match": False,
                    "unanswerable_correct": False,
                }
            retrieval_metric = compute_retrieval_recall(
                units,
                item.get("evidence_plan", []),
                k_values=(3, 5, 10, 20, 50),
                required_session_ids=item.get("answer_session_ids", []),
            )
            query = {
                "original": item.get("question", ""),
                "retrieval_query": item.get("question", ""),
                "rewrite_model": None,
            }
            if self.condition in RETRIEVED_CONDITIONS and self.retrieval is not None:
                query = retrieval_query_details(
                    record,
                    self.retrieval,
                    ranking=(
                        "production"
                        if self.condition == "end_to_end"
                        else "diagnostic"
                    ),
                )
            trace = make_item_trace(
                question_id=str(record["question_id"]),
                query=query,
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
            trace["retrieval"].update({
                "fact_recall_at_k": retrieval_metric["fact_recall_at_k"],
                "complete_chain_at_k": retrieval_metric["complete_chain_at_k"],
                "required_session_recall_at_k": retrieval_metric[
                    "required_session_recall_at_k"
                ],
            })
            return {
                "question_id": record["question_id"],
                "persona_uuid": record["persona_uuid"],
                "question_type": item.get("question_type"),
                "answerability": item.get("answerability"),
                "question": item.get("question"),
                "expected_answer": item.get("answer"),
                "answer_aliases": item.get("answer_aliases", []),
                "condition": self.condition,
                "predicted_answer": answer.text,
                "api_status": answer.api_status,
                "evaluation": evaluation,
                "retrieval_recall": retrieval_metric,
                "trace": trace,
                "usage": {
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    "total_tokens": answer.total_tokens,
                    "estimated_cost_usd": answer.estimated_cost_usd,
                },
                "latency_seconds": round(time.time() - started, 3),
                **auxiliary,
            }
        except Exception as error:
            logger.warning("{} failed: {}", record["question_id"], error)
            return _terminal_record(record, self.condition, error)

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
                    desc=self.condition, unit="item"
                ):
                    record = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(_terminal_record(record, self.condition, error))
        order = {
            str(record["question_id"]): index
            for index, record in enumerate(self.records)
        }
        results.sort(key=lambda row: order[str(row["question_id"])])
        validate_result_qids(results, list(order))
        return results


def _select_records(
    records: Sequence[Mapping[str, Any]],
    *,
    persona_filter: str | None,
    max_items: int | None,
    qid_manifest: str | None,
) -> list[dict[str, Any]]:
    selected = [
        dict(record) for record in records
        if record.get("answerability") == "answerable"
    ]
    if persona_filter:
        filters = [value.strip() for value in persona_filter.split(",") if value.strip()]
        selected = [
            record for record in selected
            if any(value in str(record["persona_uuid"]) for value in filters)
        ]
    if qid_manifest:
        payload = _read_json(qid_manifest)
        requested = payload.get("question_ids", payload)
        if not isinstance(requested, list):
            raise ValueError("Pilot QID manifest must be a list or contain question_ids")
        by_qid = {str(record["question_id"]): record for record in selected}
        missing = [qid for qid in requested if str(qid) not in by_qid]
        if missing:
            raise ValueError(f"Pilot manifest contains unavailable QIDs: {missing[:5]}")
        selected = [by_qid[str(qid)] for qid in requested]
    if max_items is not None:
        selected = selected[:max_items]
    return selected


def _failed_result(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("api_status") or "") in {"error", "timeout", "empty"}
        or bool(row.get("error"))
    )


def _load_resume_results(
    path: Path, expected_qids: Sequence[str], condition: str
) -> tuple[list[dict[str, Any]], set[str]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    validate_result_qids(rows, expected_qids)
    wrong = [
        row.get("question_id") for row in rows
        if row.get("condition") != condition
    ]
    if wrong:
        raise ValueError(
            f"Resume file contains a different condition ({wrong[:5]})"
        )
    return rows, {
        str(row["question_id"]) for row in rows if _failed_result(row)
    }


def _require_api_key(config: Mapping[str, Any], label: str) -> None:
    if config.get("api_key"):
        return
    env_name = config.get("api_key_env", "API key environment variable")
    raise ValueError(f"{label} API key is unset; set {env_name}")


def _summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_results(results)
    statuses = Counter(str(row.get("api_status") or "unknown") for row in results)
    latencies = [float(row.get("latency_seconds") or 0.0) for row in results]
    costs = [
        float(row.get("usage", {}).get("estimated_cost_usd") or 0.0)
        for row in results
        if row.get("usage", {}).get("estimated_cost_usd") is not None
    ]
    aggregate.update({
        "api_status_counts": dict(statuses),
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 4)
        if latencies else 0.0,
        "total_estimated_cost_usd": round(sum(costs), 6) if costs else None,
    })
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EVALUATION_ROOT / "configs" / "stage_decomposition.json"),
    )
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--persona-filter")
    parser.add_argument("--qid-manifest", help="Ordered pilot question-ID JSON")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--prepare-retrieval-manifest", action="store_true",
        help="Build/validate the shared top-50 manifest without answer API calls",
    )
    parser.add_argument("--force-retrieval-manifest", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Retry only terminal error/timeout/empty records in details.jsonl",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Build every selected condition context without model API calls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    canonical = load_canonical_records(config.get("canonical_data_dir"))
    records = _select_records(
        canonical,
        persona_filter=args.persona_filter,
        max_items=args.max_items,
        qid_manifest=args.qid_manifest,
    )
    if not records:
        raise SystemExit("No answerable records selected")

    retrieval: RetrievalManifest | None = None
    if args.prepare_retrieval_manifest or args.condition in RETRIEVED_CONDITIONS:
        retrieval_config = {
            "index_dir": config["native_index_dir"],
            "top_k": config.get("retrieval_top_k", 50),
            "embedding": config.get("embedding", {}),
            "production_query_manifest": config["production_query_manifest"],
        }
        retrieval = RetrievalManifest(
            config["retrieval_manifest"], retrieval_config
        )
    if args.prepare_retrieval_manifest:
        if retrieval is None:
            raise RuntimeError("Retrieval manifest initialization failed")
        full_answerable = [
            dict(record) for record in canonical
            if record.get("answerability") == "answerable"
        ]
        if [record["question_id"] for record in records] != [
            record["question_id"] for record in full_answerable
        ]:
            raise ValueError(
                "Build the shared retrieval manifest on the full 784-item set; "
                "do not combine --prepare-retrieval-manifest with subset filters"
            )
        retrieval.build(records, force=args.force_retrieval_manifest)
        print(f"Retrieval manifest ready: {retrieval.path} ({len(retrieval.entries)} items)")
        if not args.condition:
            return

    if not args.condition:
        raise ValueError("--condition is required unless only preparing retrieval")
    if args.condition in RETRIEVED_CONDITIONS:
        if retrieval is None:
            raise RuntimeError("Retrieved conditions require a retrieval manifest")
        retrieval.validate_qids([str(record["question_id"]) for record in canonical if record.get("answerability") == "answerable"])

    output_dir = Path(args.output_dir or Path(config["output_root"]) / args.condition)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "run.log", level="DEBUG")

    runner = StageDecompositionRunner(
        config, args.condition, records, output_dir, retrieval=retrieval
    )
    if args.validate_only:
        contexts: list[dict[str, Any]] = []
        for record in tqdm(records, desc="Validate contexts", unit="item"):
            units, auxiliary = runner.build_units(
                record, generate_structured=False
            )
            delivery = _pack(args.condition, units, config)
            contexts.append({
                "question_id": record["question_id"],
                "num_units": len(units),
                "num_delivered": len(delivery.get("delivered", [])),
                "num_dropped": len(delivery.get("dropped", [])),
                "drop_reasons": delivery.get("drop_reasons", {}),
                "writer_cache_keys": [
                    value.get("cache_key")
                    for value in auxiliary.get("structured_writer", [])
                ],
            })
        _write_json_atomic(output_dir / "context_validation.json", {
            "condition": args.condition,
            "num_items": len(contexts),
            "question_id_sha256": canonical_qid_sha256(
                row["question_id"] for row in contexts
            ),
            "items": contexts,
        })
        print(f"Validated {len(contexts)} contexts: {output_dir}")
        return

    _require_api_key(config["reader"], "Reader")
    if args.condition == "gold_structured_text":
        _require_api_key(config["structured_writer"], "Structured writer")

    workers = int(args.workers or config.get("workers", 10))
    existing_results: list[dict[str, Any]] | None = None
    records_to_run = records
    details_path = output_dir / "details.jsonl"
    if args.resume:
        if not details_path.is_file():
            raise FileNotFoundError(f"Resume requested but missing {details_path}")
        expected_qids = [str(record["question_id"]) for record in records]
        existing_results, failed_qids = _load_resume_results(
            details_path, expected_qids, args.condition
        )
        if not failed_qids:
            print(f"No terminal failures to resume: {details_path}")
            return
        records_to_run = [
            record for record in records
            if str(record["question_id"]) in failed_qids
        ]
        runner.records = records_to_run
        logger.info("Resuming {} failed items", len(records_to_run))

    results = runner.run(workers)
    if existing_results is not None:
        replacements = {str(row["question_id"]): row for row in results}
        results = [
            replacements.get(str(row["question_id"]), row)
            for row in existing_results
        ]
        validate_result_qids(
            results, [str(record["question_id"]) for record in records]
        )
    write_details_jsonl(output_dir, results)
    summary = _summary(results)
    _write_json_atomic(output_dir / "summary.json", summary)
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get(
            "canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"
        ),
        num_requested_items=len(records),
        evaluated_item_count=len(results),
        execution_track="stage_decomposition",
        memory_organization="controlled_evidence",
        representation=args.condition,
        index_manifest=(
            str(retrieval.path)
            if retrieval is not None and args.condition in RETRIEVED_CONDITIONS
            else None
        ),
        retriever=(
            config.get("embedding", {}).get("model_name")
            if args.condition in RETRIEVED_CONDITIONS else "gold_evidence"
        ),
        native_system_reader=None,
        chunk_unit="evidence_carrier",
        query_policy=(
            "fixed_rewrite" if args.condition == "end_to_end" else "original"
        ),
        answer_reasoner=config["reader"].get("llm_model"),
        judge=config.get("judge"),
        top_k=(config.get("retrieval_top_k", 50) if args.condition in RETRIEVED_CONDITIONS else None),
        delivery_budget={
            "condition": args.condition,
            "production_top_k": (
                int(config.get("production_top_k", 20))
                if args.condition in {"retrieved_raw_budgeted", "end_to_end"}
                else None
            ),
            "policy": {
                **_delivery_policy(args.condition, config).__dict__,
                "supported_modalities": sorted(
                    _delivery_policy(args.condition, config).supported_modalities
                ),
            },
        },
        controlled_qid_sha256=canonical_qid_sha256(
            record["question_id"] for record in records
        ),
        answer_prompt_version=ANSWER_PROMPT_VERSION,
        structured_writer_version=STRUCTURED_WRITER_VERSION,
        retrieval_manifest_sha256=(
            sha256_file(retrieval.path)
            if retrieval is not None and args.condition in RETRIEVED_CONDITIONS
            else None
        ),
        production_query_manifest=(
            config.get("production_query_manifest")
            if args.condition == "end_to_end" else None
        ),
        production_query_manifest_sha256=(
            sha256_file(config["production_query_manifest"])
            if args.condition == "end_to_end" else None
        ),
    )
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
