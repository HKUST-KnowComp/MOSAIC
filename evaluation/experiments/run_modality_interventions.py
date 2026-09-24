#!/usr/bin/env python3
"""Run carrier-level spatial, temporal, and prosodic interventions.

Each experimental case identifies one required multimodal carrier. The target
carrier is changed according to the requested condition while every other
gold evidence hop remains fixed. Cases are paired by ``intervention_id``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from collections import Counter
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
from answer_harness import ANSWER_PROMPT_VERSION, CommonAnswerHarness, ObservableChatRunner  # noqa: E402
from protocol import (  # noqa: E402
    DeliveryPolicy,
    canonical_qid_sha256,
    make_item_trace,
    materialize_openai_media_blocks,
    pack_evidence,
    sha256_file,
    sha256_json,
    write_details_jsonl,
    write_run_manifest,
)
from run_stage_decomposition import (  # noqa: E402
    TextualizedCorpus,
    _carrier_map,
    _read_json,
    _required_entries,
    _require_api_key,
    _summary,
    _unit,
    _write_json_atomic,
    build_gold_raw_evidence,
)
from protocol import load_canonical_records  # noqa: E402


CONDITIONS: dict[str, tuple[str, str]] = {
    "image_raw": ("image", "raw"),
    "image_ocr_only": ("image", "ocr_only"),
    "image_object_only": ("image", "object_only"),
    "image_spatial_description": ("image", "spatial_description"),
    "video_raw": ("video", "raw"),
    "video_sparse_ordered_frames": ("video", "sparse_ordered_frames"),
    "video_shuffled_frames": ("video", "shuffled_frames"),
    "video_ordered_event_count_description": (
        "video", "ordered_event_count_description"
    ),
    "audio_raw": ("audio", "raw"),
    "audio_asr": ("audio", "asr"),
    "audio_asr_prosody": ("audio", "asr_prosody"),
    # Causal selector sensitivity: identical ASR context with happy↔unhappy flipped.
    "audio_asr_prosody_swapped": ("audio", "asr_prosody_swapped"),
    # Same spoken words, opposite-tone TTS waveform (E7). Requires swapped_tts_manifest.
    "audio_raw_tone_swapped": ("audio", "raw_tone_swapped"),
}

PROSODY_TONE_FLIP = {
    "happy": "unhappy",
    "unhappy": "happy",
}

CHALLENGES = {
    "image": {"spatial_configuration", "fine_grained_attribute", "background_detail"},
    "video": {"action_ordering", "state_transition", "action_repetition_count"},
    "audio": {"happy", "unhappy"},
}

PAPER_CHALLENGE = {
    "spatial_configuration": "spatial_configuration",
    "fine_grained_attribute": "ocr_fine_detail",
    "background_detail": "ocr_fine_detail",
    "action_ordering": "action_ordering",
    "state_transition": "state_transition",
    "action_repetition_count": "repetition_count",
    "happy": "prosody",
    "unhappy": "prosody",
}

TRANSFORMATION_VERSION = "query-answer-blind-interventions-v1"
LEAK_AUDIT_VERSION = 2
TRANSFORMATION_PROMPTS = {
    "ocr_only": (
        "Inspect only the attached image. Transcribe all visible text exactly, "
        "including numbers and symbols. Return plain text only. Do not name "
        "objects, infer meaning, or describe positions except line order."
    ),
    "object_only": (
        "Inspect only the attached image. Return a compact JSON inventory with "
        "objects, observable attributes, and counts. Omit all visible text, "
        "relative positions, layout, and downstream interpretation."
    ),
    "spatial_description": (
        "Inspect only the attached image. Return compact JSON with scene, "
        "entities, and explicit relative spatial relations (left/right, "
        "above/below, inside/near/behind). Do not transcribe visible text and "
        "do not infer any downstream question."
    ),
    "ordered_event_count_description": (
        "Inspect only the attached video. Return compact JSON with entities, "
        "chronologically ordered events, state transitions, and repetition "
        "counts. Do not infer any downstream question or answer."
    ),
}


def _entry_challenge(entry: Mapping[str, Any]) -> str:
    modality = str(entry.get("modality") or "")
    if modality in {"image", "video"}:
        return str(entry.get("visual_gap_strategy") or "")
    if modality == "audio":
        return str(entry.get("audio_tone") or "")
    return ""


def _intervention_id(question_id: str, fact_key: str) -> str:
    return f"{question_id}::{fact_key}"


def select_cases(
    records: Sequence[Mapping[str, Any]],
    modality: str,
    *,
    challenges: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Create one case per matching required carrier in canonical order."""
    accepted = challenges or CHALLENGES[modality]
    cases: list[dict[str, Any]] = []
    for record in records:
        if record.get("answerability") != "answerable":
            continue
        for entry in _required_entries(record["item"]):
            challenge = _entry_challenge(entry)
            if entry.get("modality") != modality or challenge not in accepted:
                continue
            fact_key = str(entry["fact_key"])
            cases.append({
                "intervention_id": _intervention_id(
                    str(record["question_id"]), fact_key
                ),
                "question_id": str(record["question_id"]),
                "target_fact_key": fact_key,
                "target_modality": modality,
                "challenge": challenge,
                "paper_challenge": PAPER_CHALLENGE[challenge],
                "entry": dict(entry),
                "record": dict(record),
            })
    return cases


def _load_case_manifest(path: str) -> list[str]:
    payload = _read_json(path)
    values = payload.get("intervention_ids", payload.get("question_ids", payload))
    if not isinstance(values, list):
        raise ValueError(
            "Case manifest must be a list or contain intervention_ids/question_ids"
        )
    return [str(value) for value in values]


def _filter_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    qid_manifest: str | None,
    max_items: int | None,
) -> list[dict[str, Any]]:
    selected = [dict(case) for case in cases]
    if qid_manifest:
        requested = _load_case_manifest(qid_manifest)
        by_id = {str(case["intervention_id"]): case for case in selected}
        qid_values = {str(case["question_id"]) for case in selected}
        if all(value in by_id for value in requested):
            selected = [dict(by_id[value]) for value in requested]
        elif all(value in qid_values for value in requested):
            requested_set = set(requested)
            selected = [
                dict(case) for case in selected
                if str(case["question_id"]) in requested_set
            ]
            order = {qid: index for index, qid in enumerate(requested)}
            selected.sort(key=lambda case: (
                order[str(case["question_id"])], str(case["target_fact_key"])
            ))
        else:
            available = set(by_id) | qid_values
            missing = [value for value in requested if value not in available]
            raise ValueError(f"Case manifest contains unavailable IDs: {missing[:5]}")
    if max_items is not None:
        selected = selected[:max_items]
    return selected


class TransformationWriter:
    """Question-blind persistent writer for image/video text interventions."""

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

    def _key(self, media_path: str, variant: str) -> str:
        return sha256_json({
            "media_sha256": sha256_file(media_path),
            "variant": variant,
            "prompt": TRANSFORMATION_PROMPTS[variant],
            "version": TRANSFORMATION_VERSION,
            "model": self.config.get("llm_model"),
        })

    def _save_locked(self) -> None:
        _write_json_atomic(self.cache_path, {
            "manifest_version": 1,
            "transformation_version": TRANSFORMATION_VERSION,
            "model": self.config.get("llm_model"),
            "num_entries": len(self._entries),
            "entries": dict(sorted(self._entries.items())),
        })

    def transform(
        self,
        media_path: str,
        modality: str,
        variant: str,
        *,
        generate: bool,
    ) -> tuple[str, dict[str, Any]]:
        if variant not in TRANSFORMATION_PROMPTS:
            raise ValueError(f"No transformation prompt for {variant}")
        if not os.path.isfile(media_path):
            raise FileNotFoundError(media_path)
        key = self._key(media_path, variant)
        with self._lock:
            cached = dict(self._entries.get(key) or {})
        if cached.get("api_status") == "ok":
            return str(cached.get("description") or ""), cached
        if not generate:
            return "[TRANSFORMATION PENDING]", {
                "cache_key": key,
                "media_path": media_path,
                "media_sha256": sha256_file(media_path),
                "modality": modality,
                "variant": variant,
                "prompt_sha256": sha256_json(TRANSFORMATION_PROMPTS[variant]),
                "api_status": "pending",
                "query_blind": True,
                "answer_blind": True,
            }

        delivery = pack_evidence([{
            "chunk_id": key,
            "modality": modality,
            "media_path": media_path,
            "content": "",
        }], DeliveryPolicy(mode="all_retrieved_delivered", max_media_size_mb=None))
        delivery = materialize_openai_media_blocks(
            delivery, api_format=self.config.get("api_format")
        )
        blocks = delivery.get("provider_blocks", [])
        if not blocks:
            raise ValueError(f"Could not serialize intervention media: {media_path}")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": TRANSFORMATION_PROMPTS[variant]},
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
            "variant": variant,
            "prompt_sha256": sha256_json(TRANSFORMATION_PROMPTS[variant]),
            "description": result.text,
            "query_blind": True,
            "answer_blind": True,
            **result.to_dict(),
        }
        with self._lock:
            self._entries[key] = entry
            self._save_locked()
        if result.api_status != "ok":
            raise RuntimeError(
                result.error_message or "Intervention writer returned no text"
            )
        return result.text, entry


class SparseFrameCache:
    """Content-addressed deterministic sparse video-frame extraction."""

    VERSION = "ffmpeg-uniform-v1"

    def __init__(self, root: str | os.PathLike[str], frame_count: int = 8) -> None:
        self.root = Path(root)
        self.frame_count = max(2, int(frame_count))
        self._lock = threading.Lock()

    def _key(self, media_path: str) -> str:
        return sha256_json({
            "media_sha256": sha256_file(media_path),
            "frame_count": self.frame_count,
            "version": self.VERSION,
        })

    def _duration(self, media_path: str) -> float:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", media_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        duration = float(completed.stdout.strip())
        if duration <= 0:
            raise ValueError(f"Video duration is not positive: {media_path}")
        return duration

    def frames(self, media_path: str, *, generate: bool) -> tuple[list[str], dict[str, Any]]:
        if not os.path.isfile(media_path):
            raise FileNotFoundError(media_path)
        key = self._key(media_path)
        frame_dir = self.root / key
        paths = [str(frame_dir / f"frame_{index:02d}.jpg") for index in range(self.frame_count)]
        if all(os.path.isfile(path) for path in paths):
            return paths, {
                "cache_key": key,
                "media_sha256": sha256_file(media_path),
                "frame_count": self.frame_count,
                "extractor_version": self.VERSION,
                "api_status": "ok",
                "cached": True,
            }
        if not generate:
            # The original path is a structural stand-in only; validate-only
            # never serializes it as an image provider block.
            return [media_path] * self.frame_count, {
                "cache_key": key,
                "media_sha256": sha256_file(media_path),
                "frame_count": self.frame_count,
                "extractor_version": self.VERSION,
                "api_status": "pending",
                "cached": False,
            }

        with self._lock:
            if all(os.path.isfile(path) for path in paths):
                return paths, {
                    "cache_key": key,
                    "media_sha256": sha256_file(media_path),
                    "frame_count": self.frame_count,
                    "extractor_version": self.VERSION,
                    "api_status": "ok",
                    "cached": True,
                }
            frame_dir.mkdir(parents=True, exist_ok=True)
            duration = self._duration(media_path)
            timestamps = [
                duration * (index + 0.5) / self.frame_count
                for index in range(self.frame_count)
            ]
            for timestamp, output_path in zip(timestamps, paths):
                subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{timestamp:.6f}", "-i", media_path,
                        "-frames:v", "1", "-q:v", "2", output_path,
                    ],
                    check=True,
                    capture_output=True,
                )
        if not all(os.path.isfile(path) for path in paths):
            raise RuntimeError(f"Sparse frame extraction incomplete: {media_path}")
        return paths, {
            "cache_key": key,
            "media_sha256": sha256_file(media_path),
            "frame_count": self.frame_count,
            "extractor_version": self.VERSION,
            "api_status": "ok",
            "cached": False,
        }


def _generic_transcript(record: Mapping[str, Any], fact_key: str,
                        corpus: TextualizedCorpus) -> str:
    textualized_item = corpus.item(record)
    _, message = _carrier_map(textualized_item)[fact_key]
    mm = message.get("multimodal_content") or {}
    transcript = str(mm.get("textualized") or message.get("content") or "").strip()
    if not transcript:
        raise ValueError(f"Empty ASR transcript for {record['question_id']}:{fact_key}")
    return transcript


def _replace_target(
    units: Sequence[Mapping[str, Any]],
    fact_key: str,
    replacements: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    replaced = 0
    for unit in units:
        if fact_key in set(unit.get("fact_keys") or []):
            output.extend(dict(value) for value in replacements)
            replaced += 1
        else:
            output.append(dict(unit))
    if replaced != 1:
        raise ValueError(f"Expected one target carrier for {fact_key}, found {replaced}")
    return output


def _text_replacement(
    case: Mapping[str, Any], base_unit: Mapping[str, Any], content: str, variant: str
) -> dict[str, Any]:
    return _unit(
        chunk_id=f"intervention:{case['intervention_id']}:{variant}",
        session={
            "session_id": base_unit.get("session_id", ""),
            "timestamp": base_unit.get("timestamp", ""),
        },
        modality="text",
        content=content,
        fact_key=str(case["target_fact_key"]),
    )


def _non_target_signature(units: Sequence[Mapping[str, Any]], target: str) -> str:
    retained = [dict(unit) for unit in units if target not in set(unit.get("fact_keys") or [])]
    return sha256_json(retained)


def _target_signature(units: Sequence[Mapping[str, Any]], target: str) -> str:
    selected = [
        dict(unit) for unit in units
        if target in set(unit.get("fact_keys") or [])
    ]
    return sha256_json(selected)


def _forbidden_metadata_keys(value: Any) -> set[str]:
    forbidden = {
        "question", "answer", "expected_answer", "answer_aliases", "fact_key",
        "role", "value", "intended_answer", "generation_prompt",
    }
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in forbidden:
                found.add(str(key))
            found.update(_forbidden_metadata_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_metadata_keys(child))
    return found


class ModalityInterventionRunner:
    def __init__(self, config: Mapping[str, Any], condition: str,
                 cases: Sequence[Mapping[str, Any]], output_dir: str | os.PathLike[str]) -> None:
        self.config = dict(config)
        self.condition = condition
        self.modality, self.variant = CONDITIONS[condition]
        self.cases = [dict(case) for case in cases]
        self.output_dir = Path(output_dir)
        self.harness = CommonAnswerHarness(self.config["reader"])
        self.generic = TextualizedCorpus(self.config["generic_text_dir"])
        self.writer = TransformationWriter(
            self.config["intervention_writer"], self.config["transformation_cache"]
        )
        self.frames = SparseFrameCache(
            self.config["frame_cache"], self.config.get("sparse_frame_count", 8)
        )
        self._swapped_tts: dict[str, dict[str, Any]] | None = None

    def _tone_swapped_job(self, intervention_id: str) -> dict[str, Any]:
        if self._swapped_tts is None:
            path = Path(
                self.config.get("swapped_tts_manifest")
                or (
                    Path(self.config["output_root"])
                    / "e7_opposite_tone_tts"
                    / "tts_jobs.json"
                )
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"audio_raw_tone_swapped needs swapped_tts_manifest: {path}"
                )
            payload = _read_json(str(path))
            jobs = payload.get("by_intervention_id") or {
                str(job["intervention_id"]): job for job in payload.get("jobs") or []
            }
            self._swapped_tts = {str(key): dict(value) for key, value in jobs.items()}
        if intervention_id not in self._swapped_tts:
            raise KeyError(f"No E7 TTS job for {intervention_id}")
        return self._swapped_tts[intervention_id]

    def build_units(self, case: Mapping[str, Any], *, generate: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        record = case["record"]
        item = record["item"]
        fact_key = str(case["target_fact_key"])
        base = build_gold_raw_evidence(item)
        target_units = [
            unit for unit in base if fact_key in set(unit.get("fact_keys") or [])
        ]
        if len(target_units) != 1:
            raise ValueError(f"Target carrier join failed for {case['intervention_id']}")
        target = target_units[0]
        auxiliary: dict[str, Any] = {
            "transformation_provenance": {
                "input_fields": ["media_path", "modality", "variant"],
                "query_blind": True,
                "answer_blind": True,
            }
        }
        if self.variant == "raw":
            units = base
        elif self.variant == "raw_tone_swapped":
            job = self._tone_swapped_job(str(case["intervention_id"]))
            swapped_path = str(job["output_path"])
            if not os.path.isfile(swapped_path):
                raise FileNotFoundError(
                    f"E7 swapped TTS missing for {case['intervention_id']}: {swapped_path}"
                )
            replaced = dict(target)
            replaced["media_path"] = swapped_path
            replaced["chunk_id"] = f"intervention:{case['intervention_id']}:{self.variant}"
            units = _replace_target(base, fact_key, [replaced])
            auxiliary["transformation_provenance"]["input_fields"] = [
                "frozen_spoken_text",
                "same_voice",
                "flipped_tts_instruction",
            ]
            auxiliary["tone_swap"] = {
                "gold_tone": job.get("gold_tone"),
                "delivered_tone": job.get("delivered_tone"),
                "voice": job.get("voice"),
                "rewrite_emotional_text": False,
                "media_path": swapped_path,
            }
        elif self.variant in TRANSFORMATION_PROMPTS:
            auxiliary["transformation_provenance"]["input_fields"] = [
                "media_sha256", "modality", "variant", "fixed_prompt"
            ]
            text, writer_record = self.writer.transform(
                str(target.get("media_path") or ""), self.modality,
                self.variant, generate=generate,
            )
            units = _replace_target(
                base, fact_key,
                [_text_replacement(case, target, text, self.variant)],
            )
            auxiliary["transformation"] = writer_record
        elif self.variant in {"asr", "asr_prosody", "asr_prosody_swapped"}:
            auxiliary["transformation_provenance"]["input_fields"] = [
                "generic_asr_transcript"
            ]
            transcript = _generic_transcript(record, fact_key, self.generic)
            if self.variant in {"asr_prosody", "asr_prosody_swapped"}:
                auxiliary["transformation_provenance"]["input_fields"].append(
                    "canonical_audio_tone_annotation"
                )
                gold_tone = str(case["entry"].get("audio_tone") or "unknown")
                if self.variant == "asr_prosody_swapped":
                    if gold_tone not in PROSODY_TONE_FLIP:
                        raise ValueError(
                            f"Cannot flip prosody tone {gold_tone!r} for "
                            f"{case['intervention_id']}"
                        )
                    tone = PROSODY_TONE_FLIP[gold_tone]
                    auxiliary["transformation_provenance"]["input_fields"].append(
                        "swapped_audio_tone_annotation"
                    )
                    auxiliary["prosody_tag"] = {
                        "gold_tone": gold_tone,
                        "delivered_tone": tone,
                        "swapped": True,
                    }
                else:
                    tone = gold_tone
                    auxiliary["prosody_tag"] = {
                        "gold_tone": gold_tone,
                        "delivered_tone": tone,
                        "swapped": False,
                    }
                transcript = f"Prosody: {tone}\nTranscript: {transcript}"
            units = _replace_target(
                base, fact_key,
                [_text_replacement(case, target, transcript, self.variant)],
            )
        elif self.variant in {"sparse_ordered_frames", "shuffled_frames"}:
            auxiliary["transformation_provenance"]["input_fields"] = [
                "media_sha256", "fixed_frame_count", "deterministic_order_policy"
            ]
            frame_paths, frame_record = self.frames.frames(
                str(target.get("media_path") or ""), generate=generate
            )
            order = list(range(len(frame_paths)))
            if self.variant == "shuffled_frames":
                seed = int(hashlib.sha256(
                    str(case["intervention_id"]).encode("utf-8")
                ).hexdigest()[:16], 16)
                random.Random(seed).shuffle(order)
                if order == list(range(len(frame_paths))):
                    order = order[1:] + order[:1]
            replacements = []
            for display_index, source_index in enumerate(order, start=1):
                label = (
                    f"Sparse frame {display_index} of {len(order)}, chronological order."
                    if self.variant == "sparse_ordered_frames"
                    else f"Sparse frame {display_index} of {len(order)}, shuffled order."
                )
                replacements.append(_unit(
                    chunk_id=(
                        f"intervention:{case['intervention_id']}:{self.variant}:"
                        f"{display_index}"
                    ),
                    session={
                        "session_id": target.get("session_id", ""),
                        "timestamp": target.get("timestamp", ""),
                    },
                    modality="image",
                    content=label,
                    fact_key=fact_key,
                    media_path=frame_paths[source_index],
                ))
            units = _replace_target(base, fact_key, replacements)
            auxiliary["frame_extraction"] = {
                **frame_record,
                "display_source_order": order,
            }
        else:
            raise ValueError(f"Unsupported intervention variant: {self.variant}")

        baseline_non_target = _non_target_signature(base, fact_key)
        observed_non_target = _non_target_signature(units, fact_key)
        if baseline_non_target != observed_non_target:
            raise AssertionError(
                f"Non-target evidence changed for {case['intervention_id']}"
            )
        baseline_target = _target_signature(base, fact_key)
        observed_target = _target_signature(units, fact_key)
        changed_carriers = int(baseline_target != observed_target)
        expected_changes = 0 if self.variant == "raw" else 1
        forbidden_keys = sorted(
            _forbidden_metadata_keys(auxiliary["transformation_provenance"])
        )
        query_blind = bool(
            auxiliary["transformation_provenance"].get("query_blind")
        )
        answer_blind = bool(
            auxiliary["transformation_provenance"].get("answer_blind")
        )
        auxiliary["leak_audit"] = {
            "audit_version": LEAK_AUDIT_VERSION,
            "status": "complete",
            "intervention_id": case["intervention_id"],
            "target_fact_key": fact_key,
            "baseline_target_signature": baseline_target,
            "observed_target_signature": observed_target,
            "changed_required_carriers": changed_carriers,
            "expected_changed_required_carriers": expected_changes,
            "baseline_non_target_signature": baseline_non_target,
            "observed_non_target_signature": observed_non_target,
            "non_target_hops_identical": baseline_non_target == observed_non_target,
            "transformation_query_blind": query_blind,
            "transformation_answer_blind": answer_blind,
            "forbidden_transformation_input_keys": forbidden_keys,
            "passed": (
                changed_carriers == expected_changes
                and baseline_non_target == observed_non_target
                and query_blind
                and answer_blind
                and not forbidden_keys
            ),
        }
        return units, auxiliary

    def _delivery(self, units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        delivery = pack_evidence(
            units,
            DeliveryPolicy(
                mode="all_retrieved_delivered",
                max_images=None,
                max_audio=None,
                max_video=None,
                max_media_size_mb=None,
            ),
        )
        return materialize_openai_media_blocks(
            delivery, api_format=self.config["reader"].get("api_format")
        )

    def terminal(
        self,
        case: Mapping[str, Any],
        error: Exception | str,
        *,
        leak_audit: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = case["record"]
        item = record["item"]
        message = str(error)
        audit = dict(leak_audit or {
            "audit_version": LEAK_AUDIT_VERSION,
            "status": "incomplete",
            "intervention_id": case["intervention_id"],
            "target_fact_key": case["target_fact_key"],
            "changed_required_carriers": None,
            "expected_changed_required_carriers": (
                0 if self.variant == "raw" else 1
            ),
            "audit_error": message,
            "passed": False,
        })
        return {
            "intervention_id": case["intervention_id"],
            "question_id": case["question_id"],
            "persona_uuid": record["persona_uuid"],
            "question_type": item.get("question_type"),
            "answerability": "answerable",
            "question": item.get("question"),
            "expected_answer": item.get("answer"),
            "answer_aliases": item.get("answer_aliases", []),
            "condition": self.condition,
            "target_fact_key": case["target_fact_key"],
            "target_modality": case["target_modality"],
            "challenge": case["challenge"],
            "paper_challenge": case["paper_challenge"],
            "predicted_answer": "",
            "api_status": "error",
            "evaluation": {
                "exact_match": False,
                "alias_match": False,
                "unanswerable_correct": False,
            },
            "error": message,
            "leak_audit": audit,
            "trace": make_item_trace(
                question_id=str(case["question_id"]),
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

    def evaluate_one(self, case: Mapping[str, Any]) -> dict[str, Any]:
        started = time.time()
        record = case["record"]
        item = record["item"]
        auxiliary: dict[str, Any] = {}
        try:
            units, auxiliary = self.build_units(case, generate=True)
            if not auxiliary["leak_audit"]["passed"]:
                raise ValueError(f"Leak audit failed for {case['intervention_id']}")
            delivery = self._delivery(units)
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
            trace = make_item_trace(
                question_id=str(case["question_id"]),
                query={"original": item.get("question", ""),
                       "retrieval_query": item.get("question", "")},
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
            return {
                "intervention_id": case["intervention_id"],
                "question_id": case["question_id"],
                "persona_uuid": record["persona_uuid"],
                "question_type": item.get("question_type"),
                "answerability": "answerable",
                "question": item.get("question"),
                "expected_answer": item.get("answer"),
                "answer_aliases": item.get("answer_aliases", []),
                "condition": self.condition,
                "target_fact_key": case["target_fact_key"],
                "target_modality": case["target_modality"],
                "challenge": case["challenge"],
                "paper_challenge": case["paper_challenge"],
                "predicted_answer": answer.text,
                "api_status": answer.api_status,
                "evaluation": evaluation,
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
            logger.warning("{} failed: {}", case["intervention_id"], error)
            return self.terminal(
                case, error, leak_audit=auxiliary.get("leak_audit")
            )

    def run(self, workers: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if workers <= 1:
            for case in tqdm(self.cases, desc=self.condition, unit="carrier"):
                results.append(self.evaluate_one(case))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(self.evaluate_one, case): case
                    for case in self.cases
                }
                for future in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=self.condition, unit="carrier",
                ):
                    case = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as error:
                        results.append(self.terminal(case, error))
        order = {
            str(case["intervention_id"]): index
            for index, case in enumerate(self.cases)
        }
        results.sort(key=lambda row: order[str(row["intervention_id"])])
        _validate_intervention_ids(results, list(order))
        return results


def _validate_intervention_ids(rows: Sequence[Mapping[str, Any]],
                               expected: Sequence[str]) -> None:
    actual = [str(row.get("intervention_id") or "") for row in rows]
    if actual != list(expected) or len(actual) != len(set(actual)):
        raise ValueError("Intervention IDs are missing, duplicated, or out of order")


def _load_resume(path: Path, cases: Sequence[Mapping[str, Any]],
                 condition: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [str(case["intervention_id"]) for case in cases]
    _validate_intervention_ids(rows, expected)
    if any(row.get("condition") != condition for row in rows):
        raise ValueError("Resume file contains a different intervention condition")
    failed = {
        str(row["intervention_id"]) for row in rows
        if (
            row.get("api_status") in {"error", "timeout", "empty"}
            or row.get("error")
            or (row.get("leak_audit") or {}).get("audit_version")
            != LEAK_AUDIT_VERSION
            or not (row.get("leak_audit") or {}).get("passed")
        )
    }
    return rows, failed


def _write_leak_audit(output_dir: Path, condition: str,
                      rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    audits = [dict(row.get("leak_audit") or {}) for row in rows]
    missing = sum(not audit for audit in audits)
    incomplete = [
        audit.get("intervention_id")
        for audit in audits
        if audit and (
            audit.get("audit_version") != LEAK_AUDIT_VERSION
            or audit.get("status") != "complete"
        )
    ]
    failed = [
        audit.get("intervention_id")
        for audit in audits
        if audit and not audit.get("passed")
    ]
    payload = {
        "audit_version": LEAK_AUDIT_VERSION,
        "condition": condition,
        "num_cases": len(rows),
        "num_missing_audits": missing,
        "num_incomplete_audits": len(incomplete),
        "incomplete_intervention_ids": incomplete,
        "failed_intervention_ids": failed,
        "passed": missing == 0 and not incomplete and not failed,
        "criteria": {
            "baseline_changes_zero_and_intervention_changes_one_carrier": True,
            "all_non_target_hops_identical": True,
            "transformation_is_query_blind": True,
            "transformation_is_answer_blind": True,
        },
        "items": audits,
    }
    _write_json_atomic(output_dir / "leak_audit.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(EVALUATION_ROOT / "configs" / "modality_interventions.json"),
    )
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--challenge", action="append",
                        help="Exact canonical challenge; repeat to select several")
    parser.add_argument("--qid-manifest",
                        help="Ordered pilot question-ID or intervention-ID JSON")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--swapped-tts-manifest",
        help="E7 tts_jobs.json for audio_raw_tone_swapped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.swapped_tts_manifest:
        config["swapped_tts_manifest"] = args.swapped_tts_manifest
    modality, variant = CONDITIONS[args.condition]
    selected_challenges = set(args.challenge or CHALLENGES[modality])
    invalid = selected_challenges - CHALLENGES[modality]
    if invalid:
        raise ValueError(
            f"Invalid {modality} challenges: {sorted(invalid)}; "
            f"allowed={sorted(CHALLENGES[modality])}"
        )
    canonical = load_canonical_records(config.get("canonical_data_dir"))
    cases = _filter_cases(
        select_cases(canonical, modality, challenges=selected_challenges),
        qid_manifest=args.qid_manifest,
        max_items=args.max_items,
    )
    if not cases:
        raise SystemExit("No matching intervention carriers selected")

    output_dir = Path(
        args.output_dir or Path(config["output_root"]) / args.condition
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(output_dir / "run.log", level="DEBUG")
    runner = ModalityInterventionRunner(config, args.condition, cases, output_dir)

    if args.validate_only:
        rows = []
        for case in tqdm(cases, desc="Validate interventions", unit="carrier"):
            units, auxiliary = runner.build_units(case, generate=False)
            delivery = pack_evidence(
                units,
                DeliveryPolicy(
                    mode="all_retrieved_delivered",
                    max_images=None,
                    max_audio=None,
                    max_video=None,
                    max_media_size_mb=None,
                ),
            )
            rows.append({
                "intervention_id": case["intervention_id"],
                "question_id": case["question_id"],
                "target_fact_key": case["target_fact_key"],
                "challenge": case["challenge"],
                "num_units": len(units),
                "num_delivered": len(delivery.get("delivered", [])),
                "num_dropped": len(delivery.get("dropped", [])),
                **auxiliary,
            })
        audit = _write_leak_audit(output_dir, args.condition, rows)
        if not audit["passed"]:
            raise ValueError(f"Leak audit failed: {output_dir / 'leak_audit.json'}")
        _write_json_atomic(output_dir / "context_validation.json", {
            "condition": args.condition,
            "modality": modality,
            "variant": variant,
            "num_cases": len(rows),
            "intervention_id_sha256": canonical_qid_sha256(
                row["intervention_id"] for row in rows
            ),
            "challenge_counts": dict(Counter(row["challenge"] for row in rows)),
            "items": rows,
        })
        print(f"Validated {len(rows)} intervention cases: {output_dir}")
        return

    _require_api_key(config["reader"], "Reader")
    if variant in TRANSFORMATION_PROMPTS:
        _require_api_key(config["intervention_writer"], "Intervention writer")

    existing: list[dict[str, Any]] | None = None
    cases_to_run = cases
    details_path = output_dir / "details.jsonl"
    if args.resume:
        if not details_path.is_file():
            raise FileNotFoundError(f"Resume requested but missing {details_path}")
        existing, failed = _load_resume(details_path, cases, args.condition)
        if not failed:
            print(f"No terminal failures to resume: {details_path}")
            return
        cases_to_run = [
            case for case in cases if str(case["intervention_id"]) in failed
        ]
        runner.cases = cases_to_run

    results = runner.run(int(args.workers or config.get("workers", 10)))
    if existing is not None:
        replacements = {str(row["intervention_id"]): row for row in results}
        results = [
            replacements.get(str(row["intervention_id"]), row)
            for row in existing
        ]
        _validate_intervention_ids(
            results, [str(case["intervention_id"]) for case in cases]
        )
    write_details_jsonl(output_dir, results)
    audit = _write_leak_audit(output_dir, args.condition, results)
    summary = _summary(results)
    summary.update({
        "condition": args.condition,
        "modality": modality,
        "variant": variant,
        "num_intervention_cases": len(results),
        "num_unique_questions": len({row["question_id"] for row in results}),
        "challenge_counts": dict(Counter(row["challenge"] for row in results)),
        "leak_audit_passed": audit["passed"],
    })
    _write_json_atomic(output_dir / "summary.json", summary)
    if not audit["passed"]:
        raise ValueError(
            "Leak audit failed after resumable results were saved: "
            f"{output_dir / 'leak_audit.json'}. Correct the underlying error "
            "and rerun with --resume."
        )
    write_run_manifest(
        output_dir,
        config,
        canonical_manifest=config.get(
            "canonical_qid_manifest", "evaluation/manifests/canonical_qids.json"
        ),
        num_requested_items=len(cases),
        evaluated_item_count=len(results),
        execution_track="modality_intervention",
        memory_organization="gold_evidence_carriers",
        representation=args.condition,
        retriever="gold_evidence",
        chunk_unit="required_carrier",
        query_policy="original",
        answer_reasoner=config["reader"].get("llm_model"),
        judge=config.get("judge"),
        top_k=None,
        delivery_budget={"mode": "all_gold_carriers_delivered"},
        controlled_intervention_id_sha256=canonical_qid_sha256(
            case["intervention_id"] for case in cases
        ),
        controlled_unique_qid_sha256=canonical_qid_sha256(dict.fromkeys(
            str(case["question_id"]) for case in cases
        )),
        modality=modality,
        variant=variant,
        challenges=sorted(selected_challenges),
        answer_prompt_version=ANSWER_PROMPT_VERSION,
        transformation_version=TRANSFORMATION_VERSION,
        leak_audit="leak_audit.json",
        leak_audit_sha256=sha256_file(output_dir / "leak_audit.json"),
    )
    print(json.dumps(summary, indent=2))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
