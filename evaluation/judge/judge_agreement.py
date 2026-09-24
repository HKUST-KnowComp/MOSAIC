"""Compute cross-judge agreement between two or more judged run outputs.

Each input directory must contain ``details_judged.jsonl`` and
``summary_judged.json`` produced by ``eval_judge.py`` for the SAME
``details.jsonl`` (identical question-ID sets are enforced).

Reported per judge pair:
- exact band agreement on the 0.0/0.3/0.5/0.7/1.0 rubric,
- quadratic-weighted Cohen's kappa over the five ordinal bands,
- Spearman rank correlation over raw scores,
- per-question-type weighted kappa.

Usage:
  python evaluation/judge/judge_agreement.py RUN_DIR/judge RUN_DIR/judge_glm_5_2 \
      RUN_DIR/judge_grok_4_5 --output RUN_DIR/judge_agreement.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

BANDS = (0.0, 0.3, 0.5, 0.7, 1.0)


def _band_index(score: float) -> int:
    return min(range(len(BANDS)), key=lambda i: abs(BANDS[i] - float(score)))


def _load_judged(directory: Path) -> tuple[str, dict[str, dict]]:
    details = directory / "details_judged.jsonl"
    summary = directory / "summary_judged.json"
    if not details.is_file():
        raise FileNotFoundError(f"Missing {details}")
    label = directory.name
    if summary.is_file():
        with open(summary, encoding="utf-8") as handle:
            label = json.load(handle).get("judge_model", label)
    rows: dict[str, dict] = {}
    with open(details, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["question_id"])] = row
    return label, rows


def weighted_kappa(a: list[int], b: list[int], k: int = len(BANDS)) -> float:
    """Quadratic-weighted Cohen's kappa over ordinal category indices."""
    n = len(a)
    if n == 0:
        return float("nan")
    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[x][y] += 1.0
    row_marg = [sum(observed[i]) for i in range(k)]
    col_marg = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    denom = (k - 1) ** 2
    po = pe = 0.0
    for i in range(k):
        for j in range(k):
            weight = 1.0 - ((i - j) ** 2) / denom
            po += weight * observed[i][j] / n
            pe += weight * (row_marg[i] / n) * (col_marg[j] / n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in order[i:j + 1]:
            ranks[idx] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    mean_a = sum(ra) / len(ra)
    mean_b = sum(rb) / len(rb)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / (var_a * var_b) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("judged_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.judged_dirs) < 2:
        raise SystemExit("Need at least two judged directories")

    judges: list[tuple[str, dict[str, dict]]] = [
        _load_judged(d) for d in args.judged_dirs
    ]
    qid_sets = [set(rows) for _, rows in judges]
    if any(s != qid_sets[0] for s in qid_sets[1:]):
        raise SystemExit(
            "Question-ID sets differ between judged outputs; every judge must "
            "score the same details.jsonl"
        )

    # Items where every judge produced a valid score.
    valid_qids = sorted(
        qid
        for qid in qid_sets[0]
        if all(not rows[qid].get("judge_error") for _, rows in judges)
    )
    dropped = len(qid_sets[0]) - len(valid_qids)

    result: dict = {
        "judges": [label for label, _ in judges],
        "num_items": len(qid_sets[0]),
        "num_items_compared": len(valid_qids),
        "num_items_dropped_for_judge_error": dropped,
        "per_judge_mean_score": {},
        "pairwise": {},
        "all_judges_same_band_rate": None,
    }

    for label, rows in judges:
        scores = [float(rows[q]["judge_score"]) for q in valid_qids]
        result["per_judge_mean_score"][label] = (
            round(sum(scores) / len(scores), 4) if scores else None
        )

    same_band = 0
    for qid in valid_qids:
        bands = {_band_index(rows[qid]["judge_score"]) for _, rows in judges}
        if len(bands) == 1:
            same_band += 1
    if valid_qids:
        result["all_judges_same_band_rate"] = round(same_band / len(valid_qids), 4)

    for (label_a, rows_a), (label_b, rows_b) in combinations(judges, 2):
        scores_a = [float(rows_a[q]["judge_score"]) for q in valid_qids]
        scores_b = [float(rows_b[q]["judge_score"]) for q in valid_qids]
        bands_a = [_band_index(s) for s in scores_a]
        bands_b = [_band_index(s) for s in scores_b]
        exact = (
            sum(x == y for x, y in zip(bands_a, bands_b)) / len(valid_qids)
            if valid_qids else float("nan")
        )
        by_type: dict[str, dict] = defaultdict(lambda: {"a": [], "b": []})
        for qid, ba, bb in zip(valid_qids, bands_a, bands_b):
            qt = str(rows_a[qid].get("question_type", "unknown"))
            by_type[qt]["a"].append(ba)
            by_type[qt]["b"].append(bb)
        result["pairwise"][f"{label_a} vs {label_b}"] = {
            "exact_band_agreement": round(exact, 4),
            "quadratic_weighted_kappa": round(
                weighted_kappa(bands_a, bands_b), 4
            ),
            "spearman_rho": round(spearman(scores_a, scores_b), 4),
            "per_question_type_kappa": {
                qt: round(weighted_kappa(v["a"], v["b"]), 4)
                for qt, v in sorted(by_type.items())
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("CROSS-JUDGE AGREEMENT")
    print("=" * 72)
    print(f"Judges:            {', '.join(result['judges'])}")
    print(f"Items compared:    {result['num_items_compared']}"
          f" (dropped {dropped} with judge errors)")
    print(f"All-same-band:     {result['all_judges_same_band_rate']}")
    for pair, stats in result["pairwise"].items():
        print(f"\n{pair}")
        print(f"  exact band agreement: {stats['exact_band_agreement']}")
        print(f"  weighted kappa:       {stats['quadratic_weighted_kappa']}")
        print(f"  spearman rho:         {stats['spearman_rho']}")
    print(f"\nWritten -> {args.output}")


if __name__ == "__main__":
    main()
