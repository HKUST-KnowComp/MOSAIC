#!/usr/bin/env python3
"""Aggregate correct vs swapped gold-prosody-tag paired effects.

Primary unit: unique eligible prosody QIDs (mean over intervention cases
within a QID). Reports paired Δ, item-bootstrap 95% CI, W/T/L, and a
persona-clustered mean difference. Does not filter on ASR failure.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _load_scores(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.open():
        row = json.loads(line)
        iid = str(row["intervention_id"])
        rows[iid] = {
            "intervention_id": iid,
            "question_id": str(row["question_id"]),
            "persona_uuid": str(row.get("persona_uuid") or ""),
            "challenge": row.get("challenge"),
            "paper_challenge": row.get("paper_challenge"),
            "judge_score": float(row["judge_score"]),
            "predicted_answer": row.get("predicted_answer"),
            "expected_answer": row.get("expected_answer"),
        }
    return rows


def _qid_means(
    correct: Mapping[str, dict[str, Any]],
    swapped: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_qid: dict[str, list[float]] = defaultdict(list)
    persona: dict[str, str] = {}
    for iid in sorted(set(correct) & set(swapped)):
        c, s = correct[iid], swapped[iid]
        if c["question_id"] != s["question_id"]:
            raise ValueError(f"QID mismatch for {iid}")
        delta = c["judge_score"] - s["judge_score"]
        by_qid[c["question_id"]].append(delta)
        persona[c["question_id"]] = c["persona_uuid"]
    return [
        {
            "question_id": qid,
            "persona_uuid": persona[qid],
            "n_cases": len(deltas),
            "delta": sum(deltas) / len(deltas),
        }
        for qid, deltas in sorted(by_qid.items())
    ]


def _bootstrap_ci(
    values: list[float], *, samples: int, seed: int
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("empty values")
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    boots: list[float] = []
    n = len(values)
    for _ in range(samples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(draw) / n)
    boots.sort()
    lo = boots[int(0.025 * samples)]
    hi = boots[int(0.975 * samples)]
    return mean, lo, hi


def _persona_clustered_mean(rows: list[dict[str, Any]]) -> float:
    by_persona: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_persona[row["persona_uuid"] or row["question_id"]].append(row["delta"])
    persona_means = [sum(v) / len(v) for v in by_persona.values()]
    return sum(persona_means) / len(persona_means)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correct",
        required=True,
        help="Path to audio_asr_prosody judge/details_judged.jsonl",
    )
    parser.add_argument(
        "--swapped",
        required=True,
        help="Path to audio_asr_prosody_swapped judge/details_judged.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    correct = _load_scores(Path(args.correct))
    swapped = _load_scores(Path(args.swapped))
    common = sorted(set(correct) & set(swapped))
    if len(common) != len(correct) or len(common) != len(swapped):
        raise SystemExit(
            f"Incomplete pairing: correct={len(correct)} swapped={len(swapped)} "
            f"common={len(common)}"
        )

    case_deltas = [
        correct[iid]["judge_score"] - swapped[iid]["judge_score"] for iid in common
    ]
    qid_rows = _qid_means(correct, swapped)
    qid_deltas = [row["delta"] for row in qid_rows]
    mean, lo, hi = _bootstrap_ci(
        qid_deltas, samples=args.bootstrap_samples, seed=args.seed
    )
    wins = sum(1 for d in qid_deltas if d > 1e-12)
    ties = sum(1 for d in qid_deltas if abs(d) <= 1e-12)
    losses = sum(1 for d in qid_deltas if d < -1e-12)

    case_mean = sum(case_deltas) / len(case_deltas)
    payload = {
        "contrast": "audio_asr_prosody_correct_minus_swapped",
        "interpretation": (
            "Paired judge-score difference under gold delivery: "
            "ASR + correct gold prosody tag minus ASR + swapped tag. "
            "A positive delta supports causal selector sensitivity of the "
            "annotated prosody tag; it does not prove native-audio perception."
        ),
        "num_intervention_cases": len(common),
        "num_unique_qids": len(qid_rows),
        "num_personas": len({row["persona_uuid"] for row in qid_rows}),
        "primary_unit": "unique_qid_mean_delta",
        "mean_delta_qid": mean,
        "ci95_qid_bootstrap": [lo, hi],
        "wtl_qid": {"win": wins, "tie": ties, "loss": losses},
        "persona_clustered_mean_delta": _persona_clustered_mean(qid_rows),
        "mean_delta_intervention_case": case_mean,
        "mean_score_correct": sum(
            correct[iid]["judge_score"] for iid in common
        ) / len(common),
        "mean_score_swapped": sum(
            swapped[iid]["judge_score"] for iid in common
        ) / len(common),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
