#!/usr/bin/env python3
"""Deterministic evaluation protocol checks.

The checks intentionally use synthetic records only.  They do not import an
embedding model, make API calls, build a memory store, or require pytest.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from compare_parity_runs import compare_records
from protocol import (
    DeliveryPolicy,
    aggregate_results,
    compute_retrieval_recall,
    load_qid_manifest,
    make_item_trace,
    pack_evidence,
    retrieval_metric_aggregate,
    validate_result_qids,
)
from _common import partition_terminal_results
from experiments.answer_harness import (
    ANSWER_SYSTEM_PROMPT,
    ObservableChatRunner,
    build_answer_messages,
)
from experiments.run_stage_decomposition import (
    STRUCTURED_SCHEMAS,
    StageDecompositionRunner,
    StructuredWriter,
    _delivery_policy,
    _pack,
    build_gold_raw_evidence,
    build_gold_symbolic_facts,
    retrieval_units,
    retrieval_query_details,
)
from experiments.run_modality_interventions import (
    CONDITIONS as INTERVENTION_CONDITIONS,
    LEAK_AUDIT_VERSION,
    _load_resume as load_intervention_resume,
    _non_target_signature,
    _replace_target,
    _target_signature,
    _write_leak_audit,
    select_cases,
)
from experiments.run_fixed_reader_comparison import (
    _source_observability,
    _source_units,
)
from experiments.build_paper_results import (
    _compatible_fields,
    _decorate_rows,
    _validate_expected_delivery,
    _validate_judged_binding,
    _validate_judge_summary,
    _validate_stage_identity,
    load_runs,
    build_budget_curve,
    build_reader_replication,
    paired_effect,
    validate_pair,
)


def _plan(*facts: str) -> list[dict]:
    return [
        {"fact_key": fact, "required_for_answer": True}
        for fact in facts
    ]


class _FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def create(self, **_kwargs: object) -> object:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _fake_client(*outcomes: object) -> tuple[object, _FakeCompletions]:
    completions = _FakeCompletions(list(outcomes))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _fake_response(text: str, *, finish_reason: str = "stop") -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=3,
            total_tokens=14,
        ),
    )


def run_checks() -> dict[str, str]:
    manifest = load_qid_manifest(
        Path(__file__).resolve().parent / "manifests" / "canonical_qids.json"
    )
    canonical_qids = manifest["question_ids"]
    assert manifest["num_items"] == 923
    assert len(canonical_qids) == 923
    assert manifest["question_id_sha256"] == (
        "8a42ac2fe6ad94490d083bd83e0a817f9ae205f461f38d8b791c6edafc6636a4"
    )

    repo_root = Path(__file__).resolve().parent.parent
    react_configs = sorted(
        (repo_root / "evaluation" / "react").glob("config_eval_react*.json")
    )
    for react_path in react_configs:
        react_config = json.loads(react_path.read_text(encoding="utf-8"))
        baseline_path = repo_root / react_config["parity_baseline_config"]
        baseline_config = json.loads(baseline_path.read_text(encoding="utf-8"))
        assert Path(react_config["index_dir"]).resolve() == Path(
            baseline_config["index_dir"]
        ).resolve()
        assert react_config["canonical_qid_manifest"] == baseline_config[
            "canonical_qid_manifest"
        ]
        assert react_config["react"]["query_rewrite"] is False
        assert react_config["react"]["tool_query_rewrite"] is False

    paired = [{"question_id": question_id} for question_id in canonical_qids]
    pairing = validate_result_qids(paired, canonical_qids)
    assert pairing["num_results"] == 923
    try:
        validate_result_qids(paired[:-1], canonical_qids)
    except ValueError:
        pass
    else:
        raise AssertionError("Missing result QID was not rejected")
    reordered = list(paired)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    try:
        validate_result_qids(reordered, canonical_qids)
    except ValueError:
        pass
    else:
        raise AssertionError("Reordered result QIDs were not rejected")

    # A fact at rank 20 must not leak into earlier cutoffs.
    ranked = [
        {"chunk_id": f"c{i}", "fact_key": "late" if i == 19 else None}
        for i in range(20)
    ]
    metric = compute_retrieval_recall(ranked, _plan("late"))
    assert metric["fact_recall_at_k"]["1"] == 0.0
    assert metric["fact_recall_at_k"]["3"] == 0.0
    assert metric["fact_recall_at_k"]["10"] == 0.0
    assert metric["fact_recall_at_k"]["20"] == 1.0

    # Multiple facts in one chunk count at that chunk's rank.
    metric = compute_retrieval_recall(
        [{"chunk_id": "s1", "fact_keys": ["a", "b"]}], _plan("a", "b")
    )
    assert metric["fact_recall_at_k"]["1"] == 1.0
    assert metric["complete_chain_at_k"]["1"] is True

    # Duplicate keys do not inflate the denominator or numerator.
    metric = compute_retrieval_recall(
        [{"chunk_id": "s1", "fact_key": "a"}, {"chunk_id": "s2", "fact_key": "a"}],
        _plan("a", "b"),
    )
    assert metric["fact_recall_at_k"]["3"] == 0.5
    assert metric["complete_chain_at_k"]["3"] is False

    # Empty required facts are explicitly not applicable.
    metric = compute_retrieval_recall(
        [{"chunk_id": "s1", "fact_key": "noise"}],
        [{"fact_key": "noise", "required_for_answer": False}],
    )
    assert metric["status"] == "not_applicable"
    assert metric["fact_recall_at_k"]["1"] == "not_applicable"

    # Delivery caps and reasons are deterministic.
    units = [
        {"chunk_id": "i1", "modality": "image", "media_path": __file__, "fact_key": "a"},
        {"chunk_id": "i2", "modality": "image", "media_path": __file__, "fact_key": "b"},
        {"chunk_id": "a1", "modality": "audio", "media_path": __file__, "fact_key": "c"},
        {"chunk_id": "missing", "modality": "video", "media_path": "/does/not/exist", "fact_key": "d"},
    ]
    packed = pack_evidence(
        units,
        DeliveryPolicy(
            mode="production_budget",
            max_images=1,
            max_audio=1,
            max_video=1,
            max_media_size_mb=20,
        ),
    )
    assert [u["chunk_id"] for u in packed["delivered"]] == ["i1", "a1"]
    assert [u["chunk_id"] for u in packed["dropped"]] == ["i2", "missing"]
    assert packed["drop_reasons"] == {"media_count_cap": 1, "missing_file": 1}

    # Aggregation must not turn an omitted retrieval metric into zero-valued
    # eligible evidence, and answer aggregation keeps refusals separate.
    item_results = [
        {
            "question_id": "q1",
            "answerability": "answerable",
            "evaluation": {"alias_match": True},
            "retrieval_recall": compute_retrieval_recall(
                [{"fact_key": "a"}], _plan("a")
            ),
        },
        {
            "question_id": "q2",
            "answerability": "unanswerable",
            "evaluation": {"unanswerable_correct": True},
            "retrieval_recall": compute_retrieval_recall([], []),
        },
    ]
    answer_metrics = aggregate_results(item_results)
    assert answer_metrics["correct_answerable"] == 1
    assert answer_metrics["correct_unanswerable"] == 1
    retrieval_metrics = retrieval_metric_aggregate(item_results)
    assert retrieval_metrics["num_items_with_required_facts"] == 1
    ok_rows, failed_rows = partition_terminal_results([
        {"api_status": "ok"},
        {"api_status": "empty"},
        {"api_status": "timeout"},
        {"api_status": "error", "error": "failure"},
    ])
    assert len(ok_rows) == 1
    assert [row["api_status"] for row in failed_rows] == [
        "empty", "timeout", "error",
    ]

    trace = make_item_trace(
        question_id="q1",
        query={"original": "question", "retrieval_query": "query"},
        ranked=[{"chunk_id": "c1", "fact_key": "a"}],
        evidence_plan=_plan("a"),
        answer={"predicted": "answer", "api_status": "ok", "forced": False},
    )
    assert set(trace) == {"question_id", "query", "retrieval", "delivery", "answer", "react"}
    assert trace["retrieval"]["fact_recall_at_k"]["1"] == 1.0
    assert trace["answer"]["api_status"] == "ok"

    parity_row = {
        "question_id": "q1",
        "api_status": "ok",
        "predicted_answer": "answer",
        "trace": trace,
    }
    parity = compare_records([parity_row], [dict(parity_row)])
    assert parity["passed"] is True
    changed = dict(parity_row)
    changed["predicted_answer"] = "different"
    parity = compare_records([parity_row], [changed])
    assert parity["passed"] is False

    synthetic_item = {
        "question_id": "q-controlled",
        "answer": "hidden answer",
        "answer_session_ids": ["s1"],
        "evidence_plan": [{
            "fact_key": "f1",
            "fact_statement": "A clock is visible behind the object.",
            "value": "8:15 AM",
            "modality": "image",
            "required_for_answer": True,
            "role": "ANSWER",
            "hop_index": 1,
        }],
        "haystack_sessions": [{
            "session_id": "s1",
            "timestamp": "2026/01/01",
            "messages": [{
                "role": "user",
                "content": "Here is the photo.",
                "fact_key": "f1",
                "multimodal_content": {
                    "modality": "image",
                    "media_path": __file__,
                    "natural_utterance": "Here is the photo.",
                },
            }],
        }],
    }
    symbolic_units = build_gold_symbolic_facts(synthetic_item)
    symbolic_context = symbolic_units[0]["content"]
    assert "f1" not in symbolic_context
    assert "ANSWER" not in symbolic_context
    assert symbolic_units[0]["fact_keys"] == ["f1"]
    raw_units = build_gold_raw_evidence(synthetic_item)
    assert raw_units[0]["media_path"] == __file__
    assert raw_units[0]["content"] == "Here is the photo."

    messages_a = build_answer_messages("Question A?", symbolic_units, [])
    messages_b = build_answer_messages("Question B?", symbolic_units, [])
    assert messages_a[0]["content"] == messages_b[0]["content"] == ANSWER_SYSTEM_PROMPT
    assert "Question A?" in messages_a[1]["content"][-1]["text"]
    assert "Question B?" in messages_b[1]["content"][-1]["text"]
    messages_ku = build_answer_messages(
        "What changed?", symbolic_units, [], question_type="knowledge_update",
    )
    assert "from <old> to <new>" in messages_ku[0]["content"]
    assert messages_ku[0]["content"] != ANSWER_SYSTEM_PROMPT

    chat_config = {
        "llm_model": "fake-model",
        "max_attempts": 2,
        "retry_backoff": 0,
        "input_cost_per_million_tokens": 1.0,
        "output_cost_per_million_tokens": 2.0,
    }
    client, calls = _fake_client(_fake_response("8:15 AM"))
    answer = ObservableChatRunner(chat_config, client=client).complete(messages_a)
    assert answer.api_status == "ok"
    assert answer.text == "8:15 AM"
    assert answer.attempts == 1
    assert answer.input_tokens == 11
    assert answer.output_tokens == 3
    assert answer.total_tokens == 14
    assert answer.estimated_cost_usd == 0.000017
    assert calls.calls == 1

    client, calls = _fake_client(_fake_response(""), _fake_response("recovered"))
    answer = ObservableChatRunner(chat_config, client=client).complete(messages_a)
    assert answer.api_status == "ok"
    assert answer.text == "recovered"
    assert answer.attempts == 2
    assert calls.calls == 2

    client, _ = _fake_client(RuntimeError("first failure"), _fake_response(""))
    answer = ObservableChatRunner(chat_config, client=client).complete(messages_a)
    assert answer.api_status == "empty"
    assert answer.error_class is None

    class FakeTimeoutError(Exception):
        pass

    client, calls = _fake_client(FakeTimeoutError("slow"), FakeTimeoutError("slow"))
    answer = ObservableChatRunner(chat_config, client=client).complete(messages_a)
    assert answer.api_status == "timeout"
    assert answer.error_class == "FakeTimeoutError"
    assert answer.attempts == 2
    assert calls.calls == 2

    with tempfile.TemporaryDirectory() as temp_dir:
        stage_runner = StageDecompositionRunner(
            {
                "reader": {"llm_model": "unused"},
                "generic_text_dir": temp_dir,
                "structured_writer": {"llm_model": "unused"},
                "structured_cache": str(Path(temp_dir) / "structured.json"),
                "native_index_dir": str(Path(temp_dir) / "missing-index"),
                "retrieval_manifest": str(Path(temp_dir) / "retrieval.json"),
            },
            "gold_symbolic_facts",
            [],
            temp_dir,
        )
        assert stage_runner.retrieval is None
        frozen = SimpleNamespace(entries={
            "q-controlled": {
                "original_query": "What time is shown?",
                "diagnostic_ranked": [
                    {"chunk_id": f"diagnostic-{index}", "modality": "text"}
                    for index in range(50)
                ],
                "production_ranked": [
                    {"chunk_id": f"production-{index}", "modality": "text"}
                    for index in range(50)
                ],
            }
        })
        controlled_record = {
            "question_id": "q-controlled",
            "persona_uuid": "p1",
            "answerability": "answerable",
            "item": {
                **synthetic_item,
                "question": "What time is shown?",
            },
        }
        stage_config = {
            "reader": {"llm_model": "unused"},
            "generic_text_dir": temp_dir,
            "structured_writer": {"llm_model": "unused"},
            "structured_cache": str(Path(temp_dir) / "structured.json"),
            "production_top_k": 20,
        }
        all_runner = StageDecompositionRunner(
            stage_config, "retrieved_raw_all_delivered",
            [controlled_record], temp_dir, retrieval=frozen,
        )
        budget_runner = StageDecompositionRunner(
            stage_config, "retrieved_raw_budgeted",
            [controlled_record], temp_dir, retrieval=frozen,
        )
        production_runner = StageDecompositionRunner(
            stage_config, "end_to_end",
            [controlled_record], temp_dir, retrieval=frozen,
        )
        all_units, _ = all_runner.build_units(controlled_record)
        budget_units, _ = budget_runner.build_units(controlled_record)
        production_units, _ = production_runner.build_units(controlled_record)
        assert len(all_units) == 50
        assert len(budget_units) == len(production_units) == 50
        assert len(_pack("retrieved_raw_budgeted", budget_units, stage_config)[
            "delivered"
        ]) == 20
        assert len(_pack("end_to_end", production_units, stage_config)[
            "delivered"
        ]) == 20
        assert budget_units[0]["chunk_id"] == "diagnostic-0"
        assert production_units[0]["chunk_id"] == "production-0"
        assert retrieval_units(
            controlled_record, frozen, ranking="production"
        )[0]["chunk_id"] == "production-0"
        frozen.entries["q-controlled"]["production_retrieval_query"] = (
            "rewritten time query"
        )
        frozen.entries["q-controlled"]["production_rewrite_model"] = "rewriter"
        query = retrieval_query_details(
            controlled_record, frozen, ranking="production"
        )
        assert query["retrieval_query"] == "rewritten time query"
        assert query["rewrite_model"] == "rewriter"

        writer_config = {"llm_model": "writer-model"}
        first_key = StructuredWriter.cache_key(__file__, "image", writer_config)
        second_key = StructuredWriter.cache_key(__file__, "image", writer_config)
        assert first_key == second_key
        _, writer_record = StructuredWriter(
            writer_config, Path(temp_dir) / "writer.json"
        ).inspect(__file__, "image")
        serialized = json.dumps(writer_record, sort_keys=True)
        assert writer_record["api_status"] == "pending"
        assert "Question A" not in serialized
        assert "hidden answer" not in serialized
        assert "fact_key" not in serialized
        assert set(STRUCTURED_SCHEMAS) == {"image", "video", "audio"}

        cache_path = Path(temp_dir) / "retry_writer.json"
        retry_config = {
            "llm_model": "writer-model",
            "max_attempts": 1,
            "retry_backoff": 0,
        }
        failed_client, failed_calls = _fake_client(RuntimeError("transient"))
        writer = StructuredWriter(retry_config, cache_path)
        writer.chat = ObservableChatRunner(retry_config, client=failed_client)
        try:
            writer.describe(__file__, "image")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Structured writer failure was not surfaced")
        assert failed_calls.calls == 1

        recovered_client, recovered_calls = _fake_client(
            _fake_response('{"scene":"clock"}')
        )
        writer.chat = ObservableChatRunner(retry_config, client=recovered_client)
        description, recovered = writer.describe(__file__, "image")
        assert description == '{"scene":"clock"}'
        assert recovered["api_status"] == "ok"
        assert recovered_calls.calls == 1

        unused_client, unused_calls = _fake_client(RuntimeError("must not run"))
        cached_writer = StructuredWriter(retry_config, cache_path)
        cached_writer.chat = ObservableChatRunner(retry_config, client=unused_client)
        description, cached = cached_writer.describe(__file__, "image")
        assert description == '{"scene":"clock"}'
        assert cached["api_status"] == "ok"
        assert unused_calls.calls == 0

    all_policy = _delivery_policy("retrieved_raw_all_delivered", {})
    capped_all_policy = _delivery_policy(
        "retrieved_raw_all_delivered",
        {"provider_delivery_caps": {"max_video": 10}},
    )
    budget_policy = _delivery_policy(
        "retrieved_raw_budgeted",
        {"production_delivery": {"max_images": 1, "max_audio": 2, "max_video": 3}},
    )
    assert all_policy.max_images is None
    assert all_policy.max_video is None
    assert capped_all_policy.mode == "all_retrieved_delivered"
    assert capped_all_policy.max_video == 10
    assert capped_all_policy.max_images is None
    assert budget_policy.max_images == 1
    assert budget_policy.max_audio == 2
    assert budget_policy.max_video == 3

    intervention_record = {
        "question_id": "q-intervention",
        "answerability": "answerable",
        "item": {
            "question_id": "q-intervention",
            "evidence_plan": [
                {
                    "fact_key": "image_fact",
                    "modality": "image",
                    "visual_gap_strategy": "spatial_configuration",
                    "required_for_answer": True,
                    "available_at_query": True,
                },
                {
                    "fact_key": "audio_fact",
                    "modality": "audio",
                    "audio_tone": "happy",
                    "required_for_answer": True,
                    "available_at_query": True,
                },
            ],
        },
    }
    cases = select_cases([intervention_record], "image")
    assert [case["intervention_id"] for case in cases] == [
        "q-intervention::image_fact"
    ]
    assert cases[0]["paper_challenge"] == "spatial_configuration"
    assert len(INTERVENTION_CONDITIONS) == 12
    original_units = [
        {"chunk_id": "image", "fact_keys": ["image_fact"], "content": "raw"},
        {"chunk_id": "audio", "fact_keys": ["audio_fact"], "content": "fixed"},
    ]
    replacement = {
        "chunk_id": "image-text",
        "fact_keys": ["image_fact"],
        "content": "transformed",
    }
    changed_units = _replace_target(
        original_units, "image_fact", [replacement]
    )
    assert changed_units[0] == replacement
    assert _non_target_signature(
        original_units, "image_fact"
    ) == _non_target_signature(changed_units, "image_fact")
    assert _target_signature(
        original_units, "image_fact"
    ) != _target_signature(changed_units, "image_fact")

    with tempfile.TemporaryDirectory() as temp_dir:
        incomplete_row = {
            "intervention_id": "q-intervention::image_fact",
            "condition": "image_raw",
            "api_status": "error",
            "error": "synthetic context failure",
            "leak_audit": {
                "audit_version": LEAK_AUDIT_VERSION,
                "status": "incomplete",
                "intervention_id": "q-intervention::image_fact",
                "passed": False,
            },
        }
        details_path = Path(temp_dir) / "details.jsonl"
        details_path.write_text(
            json.dumps(incomplete_row) + "\n", encoding="utf-8"
        )
        _, failed = load_intervention_resume(
            details_path,
            [{"intervention_id": "q-intervention::image_fact"}],
            "image_raw",
        )
        assert failed == {"q-intervention::image_fact"}
        audit = _write_leak_audit(
            Path(temp_dir), "image_raw", [incomplete_row]
        )
        assert audit["passed"] is False
        assert audit["num_incomplete_audits"] == 1

    flat_units = _source_units({
        "retrieved": [{
            "chunk_id": "c1",
            "session_id": "s1",
            "modality": "text",
            "content": "memory",
            "fact_key": "f1",
        }]
    })
    memoryos_units = _source_units({
        "memory_read_result": {
            "retrieved_units": [{
                "chunk_id": "p1",
                "modality": "text",
                "content": "page",
                "memory_tier": "memory_page",
            }]
        }
    })
    assert flat_units[0]["fact_keys"] == ["f1"]
    assert memoryos_units[0]["memory_tier"] == "memory_page"
    amem_observability = _source_observability(
        {"memory_read_result": {"observability": {
            "write_provenance": True,
            "retrieval_ranks": True,
            "delivery_manifest": True,
        }}},
        {"observability": {"write_provenance": False}},
    )
    assert amem_observability == {
        "write_provenance": True,
        "retrieval_ranks": True,
        "delivery_manifest": True,
    }
    observability = _source_observability(
        {"memory_read_result": {"observability": {
            "write_provenance": False,
            "retrieval_ranks": True,
        }}},
        {},
    )
    assert observability == {
        "write_provenance": False,
        "retrieval_ranks": True,
        "delivery_manifest": True,
    }

    left = [
        {"question_id": "q1", "judge_score": 1.0, "persona_uuid": "p1"},
        {"question_id": "q2", "judge_score": 0.5, "persona_uuid": "p1"},
        {"question_id": "q3", "judge_score": 0.0, "persona_uuid": "p2"},
    ]
    right = [
        {"question_id": "q1", "judge_score": 0.5, "persona_uuid": "p1"},
        {"question_id": "q2", "judge_score": 0.5, "persona_uuid": "p1"},
        {"question_id": "q3", "judge_score": 1.0, "persona_uuid": "p2"},
    ]
    assert validate_pair(left, right, id_field="question_id") == ["q1", "q2", "q3"]
    effect = paired_effect(
        left, right, id_field="question_id", samples=100, seed=64,
        label="synthetic",
    )
    assert effect["mean_paired_effect_left_minus_right"] == -0.166667
    assert (effect["wins"], effect["ties"], effect["losses"]) == (1, 1, 1)
    assert effect["num_personas"] == 2

    def synthetic_run(
        rows: list[dict], *, top_k: int | None = None
    ) -> dict:
        return {
            "rows": rows,
            "id_field": "question_id",
            "manifest": {
                "top_k": 50,
                "delivery_budget": {"production_top_k": top_k},
                "answer_prompt_version": "prompt-v1",
            },
            "config": {},
            "judge_model": "judge-v1",
        }

    all_delivered_rows = [
        {
            "question_id": "q1", "persona_uuid": "p1",
            "judge_score": 1.0,
            "trace": {"delivery": {"complete_chain_covered": True}},
        },
        {
            "question_id": "q2", "persona_uuid": "p2",
            "judge_score": 0.5,
            "trace": {"delivery": {"complete_chain_covered": True}},
        },
    ]
    budget_rows = [
        {
            "question_id": "q1", "persona_uuid": "p1",
            "judge_score": 0.5,
            "trace": {"delivery": {"complete_chain_covered": False}},
        },
        {
            "question_id": "q2", "persona_uuid": "p2",
            "judge_score": 0.5,
            "trace": {"delivery": {"complete_chain_covered": True}},
        },
    ]
    curve = build_budget_curve(
        {"retrieved_raw_all_delivered": synthetic_run(all_delivered_rows)},
        {"top10_media3": synthetic_run(budget_rows, top_k=10)},
        samples=100,
        seed=64,
        allow_incomplete=False,
    )
    assert curve["budget_points"]["top10_media3"]["production_top_k"] == 10
    assert curve["budget_points"]["top10_media3"][
        "complete_chain_delivery_rate"
    ] == 0.5
    assert curve["budget_points"]["top10_media3"]["delivery_loss"][
        "mean_paired_effect_left_minus_right"
    ] == 0.25

    def scored_rows(first: float, second: float) -> list[dict]:
        return [
            {
                "question_id": "q1", "persona_uuid": "p1",
                "judge_score": first,
                "trace": {
                    "query": {"retrieval_query": "q1"},
                    "retrieval": {"ranked": [{"chunk_id": "c1"}]},
                    "delivery": {
                        "delivered": [{"chunk_id": "c1"}],
                        "dropped": [], "failed": [],
                    },
                },
            },
            {
                "question_id": "q2", "persona_uuid": "p2",
                "judge_score": second,
                "trace": {
                    "query": {"retrieval_query": "q2"},
                    "retrieval": {"ranked": [{"chunk_id": "c2"}]},
                    "delivery": {
                        "delivered": [{"chunk_id": "c2"}],
                        "dropped": [], "failed": [],
                    },
                },
            },
        ]

    primary_reader = {
        "a": synthetic_run(scored_rows(1.0, 1.0)),
        "b": synthetic_run(scored_rows(0.0, 0.0)),
        "c": synthetic_run(scored_rows(0.5, 0.5)),
    }
    secondary_reader = {
        "a": synthetic_run(scored_rows(1.0, 1.0)),
        "b": synthetic_run(scored_rows(0.0, 0.0)),
        "c": synthetic_run(scored_rows(0.2, 0.2)),
    }
    replication = build_reader_replication(
        primary_reader,
        secondary_reader,
        (
            ("perception_cost", "a", "b"),
            ("second_contrast", "a", "c"),
        ),
        samples=100,
        seed=64,
        label="synthetic_reader",
        allow_incomplete=False,
    )
    assert replication["sign_agreement_count"] == 2
    assert replication["contrast_rank_spearman"] == 1.0
    assert replication["secondary_perception_cost"] == 1.0

    canonical_info = {
        "q1": {
            "persona_uuid": "p1",
            "question_type": "information_extraction",
            "answerability": "answerable",
            "required_modalities": ["text"],
            "challenges": [],
            "chain_length": 1,
        }
    }
    judged_row = {
        "question_id": "q1",
        "persona_uuid": "p1",
        "question_type": "information_extraction",
        "answerability": "answerable",
        "api_status": "ok",
        "judge_score": 1.0,
        "judge_error": False,
        "predicted_answer": "answer",
    }
    decorated = _decorate_rows([judged_row], canonical_info, source="synthetic")
    assert decorated[0]["chain_length"] == 1
    try:
        _decorate_rows(
            [{**judged_row, "api_status": "error"}],
            canonical_info,
            source="synthetic",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Non-ok inference row was not rejected")
    try:
        _decorate_rows(
            [{**judged_row, "persona_uuid": "wrong"}],
            canonical_info,
            source="synthetic",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Canonical row metadata mismatch was not rejected")
    _validate_judge_summary(
        {"num_items": 1, "errors": 0, "judge_model": "judge"},
        decorated,
        label="synthetic",
    )
    _validate_judged_binding(
        [judged_row], [dict(judged_row)],
        id_field="question_id", label="synthetic",
    )
    _validate_expected_delivery(
        {
            "expected_delivery_budget": {
                "production_top_k": 10,
                "max_images": 3,
                "max_audio": 3,
                "max_video": 3,
            }
        },
        {
            "delivery_budget": {
                "production_top_k": 10,
                "policy": {
                    "max_images": 3, "max_audio": 3, "max_video": 3,
                },
            }
        },
        label="synthetic",
    )
    _validate_stage_identity(
        "stage_budget_sweep",
        "top10_media3",
        {
            "representation": "retrieved_raw_budgeted",
            "query_policy": "original",
            "delivery_budget": {"condition": "retrieved_raw_budgeted"},
        },
    )
    try:
        _validate_judge_summary(
            {"num_items": 2, "errors": 0, "judge_model": "judge"},
            decorated,
            label="synthetic",
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Judge summary count mismatch was not rejected")
    for group in (
        "stage_decomposition", "modality_interventions", "fixed_reader"
    ):
        assert "config_sha256" in _compatible_fields(group)

    with tempfile.TemporaryDirectory() as temp_dir:
        result_dir = Path(temp_dir) / "fixed"
        judge_dir = result_dir / "judge"
        judge_dir.mkdir(parents=True)
        (result_dir / "details.jsonl").write_text(
            json.dumps(judged_row) + "\n", encoding="utf-8"
        )
        (judge_dir / "details_judged.jsonl").write_text(
            json.dumps(judged_row) + "\n", encoding="utf-8"
        )
        (judge_dir / "summary_judged.json").write_text(
            json.dumps({
                "num_items": 1,
                "errors": 0,
                "judge_model": "judge",
            }),
            encoding="utf-8",
        )
        (result_dir / "run_manifest.json").write_text(
            json.dumps({
                "canonical_qid_sha256": "canonical",
                "evaluated_item_count": 1,
                "controlled_qid_sha256": "controlled",
                "answer_reasoner": "reader",
                "answer_prompt_version": "prompt",
                "judge": {"llm_model": "judge"},
                "top_k": 20,
                "delivery_budget": {"mode": "production_budget"},
                "config_sha256": "config",
            }),
            encoding="utf-8",
        )
        loaded, validation = load_runs(
            {
                "groups": {
                    "fixed_reader": {
                        "population": "synthetic_subset",
                        "id_field": "question_id",
                        "expected_count": 1,
                        "runs": {
                            "synthetic": {"result_dir": str(result_dir)}
                        },
                    }
                }
            },
            canonical_info,
            {
                "question_ids": ["q1"],
                "question_id_sha256": "canonical",
            },
            allow_incomplete=False,
        )
        assert list(loaded["fixed_reader"]) == ["synthetic"]
        assert validation["passed"] is True

    return {
        "canonical_manifest_923": "ok",
        "ordered_qid_pairing_and_result_count": "ok",
        "react_parity_config_pairing": "ok",
        "retrieval_rank_sensitive": "ok",
        "duplicate_fact_keys": "ok",
        "empty_required_facts": "ok",
        "delivery_caps_and_drop_reasons": "ok",
        "aggregation_invariants": "ok",
        "trace_schema": "ok",
        "parity_comparator": "ok",
        "stage_decomposition_answer_blindness": "ok",
        "stage_decomposition_prompt_invariance": "ok",
        "observable_chat_retry_usage_and_status": "ok",
        "structured_writer_cache_and_schema": "ok",
        "structured_writer_failed_cache_retry": "ok",
        "stage_gold_conditions_do_not_require_retrieval_state": "ok",
        "stage_decomposition_delivery_modes": "ok",
        "stage_dual_rankings_and_budget_cutoffs": "ok",
        "modality_intervention_pairing_and_isolation": "ok",
        "modality_intervention_measured_audit_and_resume": "ok",
        "fixed_reader_source_normalization_and_observability": "ok",
        "terminal_api_status_partitioning": "ok",
        "paper_result_pairing_and_bootstrap": "ok",
        "paper_result_budget_curve": "ok",
        "paper_result_reader_replication": "ok",
        "paper_result_fail_closed_input_gates": "ok",
        "paper_result_nonempty_controlled_load": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    args = parser.parse_args()
    result = run_checks()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for name, status in result.items():
            print(f"{name}: {status}")


if __name__ == "__main__":
    main()
