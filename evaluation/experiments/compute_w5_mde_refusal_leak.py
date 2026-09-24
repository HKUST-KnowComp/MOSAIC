#!/usr/bin/env python3
"""W5 diagnostics: QID-level video MDEs + privileged vs observable refusal tests."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

Z80 = 1.959963984540054 + 0.8416212335729143
MOD_PAT = re.compile(
    r"\b(photo|image|picture|video|clip|voice|audio|sounded|sounding|recording)\b",
    re.I,
)


def _expand_data_path(value: str) -> str:
    token = "${MOSAIC_DATA_ROOT}"
    if token not in value:
        return value
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        raise SystemExit("Set MOSAIC_DATA_ROOT to the released data directory.")
    return value.replace(token, root)


def load_judged(interv_root: Path, run: str) -> list[dict]:
    path = interv_root / run / "judge" / "details_judged.jsonl"
    return [json.loads(line) for line in path.open()]


def mde_from_deltas(deltas: list[float]) -> dict:
    n = len(deltas)
    mean = sum(deltas) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in deltas) / (n - 1))
    se = sd / math.sqrt(n)
    mde = Z80 * se
    return {
        "n": n,
        "mean_delta": round(mean, 6),
        "sd_delta": round(sd, 6),
        "se_delta": round(se, 6),
        "mde_80_power_alpha05": round(mde, 6),
        "mde_points": round(mde * 100, 1),
    }


def paired_mde(
    interv_root: Path,
    left_run: str,
    right_run: str,
    challenge: str,
    *,
    unit: str,
) -> dict:
    left = {
        row["intervention_id"]: row
        for row in load_judged(interv_root, left_run)
        if not row.get("judge_error")
    }
    right = {
        row["intervention_id"]: row
        for row in load_judged(interv_root, right_run)
        if not row.get("judge_error")
    }
    by_unit: dict[str, list[float]] = defaultdict(list)
    for iid in sorted(set(left) & set(right)):
        ch = left[iid].get("paper_challenge") or left[iid].get("challenge")
        if ch != challenge:
            continue
        key = left[iid]["question_id"] if unit == "qid" else iid
        delta = float(left[iid]["judge_score"]) - float(right[iid]["judge_score"])
        by_unit[key].append(delta)
    deltas = [sum(vals) / len(vals) for vals in by_unit.values()]
    return mde_from_deltas(deltas)


def load_rows(data_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(data_dir.glob("stage2_with_media_*.json")):
        persona = path.stem.replace("stage2_with_media_", "")
        items = json.loads(path.read_text())
        if isinstance(items, dict):
            items = items.get("items") or items.get("questions") or [items]
        for item in items:
            if not isinstance(item, dict) or "question_id" not in item:
                continue
            question = item.get("question") or ""
            hay = item.get("haystack_sessions") or []
            n_sess = len(hay) if hay else len(item.get("haystack_session_ids") or [])
            n_img = n_vid = n_aud = n_msg = 0
            dates: set[str] = set()
            for session in hay:
                if not isinstance(session, dict):
                    continue
                messages = session.get("messages") or []
                if isinstance(messages, list):
                    n_msg += len(messages)
                    for message in messages:
                        if not isinstance(message, dict):
                            continue
                        mm = message.get("multimodal_content") or {}
                        modality = str(mm.get("modality") or mm.get("type") or "").lower()
                        if modality == "image":
                            n_img += 1
                        elif modality == "video":
                            n_vid += 1
                        elif modality == "audio":
                            n_aud += 1
                if session.get("date"):
                    dates.add(str(session["date"]))
            entries = [
                entry
                for entry in item.get("evidence_plan", [])
                if entry.get("required_for_answer")
                and entry.get("available_at_query", True)
            ]
            rows.append(
                {
                    "persona": persona,
                    "y": 1 if item.get("answerability") == "answerable" else 0,
                    "q": question,
                    "qtype": item.get("question_type") or "unknown",
                    "n_hops": len(entries),
                    "n_ans_sess": len(item.get("answer_session_ids") or []),
                    "struct": [
                        len(question),
                        len(question.split()),
                        int(bool(MOD_PAT.search(question))),
                        n_sess,
                        n_msg,
                        n_img,
                        n_vid,
                        n_aud,
                        n_img + n_vid + n_aud,
                        len(dates),
                    ],
                }
            )
    return rows


def eval_auc(x, y, groups, *, n_boot: int = 1000) -> dict:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260802)
    proba = cross_val_predict(
        LogisticRegression(max_iter=5000, class_weight="balanced"),
        x,
        y,
        cv=cv,
        groups=groups,
        method="predict_proba",
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    rng = np.random.default_rng(0)
    boots: list[float] = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[sample])) < 2:
            continue
        boots.append(float(roc_auc_score(y[sample], proba[sample])))
    return {
        "auc": auc,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "cv": "StratifiedGroupKFold_persona",
    }


def refusal_block(rows: list[dict]) -> dict:
    def run(subset: list[dict]) -> dict:
        y = np.array([row["y"] for row in subset])
        groups = np.array([row["persona"] for row in subset])
        questions = [row["q"] for row in subset]
        x_struct = np.array([row["struct"] for row in subset], dtype=float)
        x_priv = np.array(
            [[row["n_hops"], row["n_ans_sess"]] for row in subset],
            dtype=float,
        )
        x_tfidf = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2
        ).fit_transform(questions)
        return {
            "n": len(subset),
            "n_answerable": int(y.sum()),
            "n_refusal": int((1 - y).sum()),
            "question_tfidf": eval_auc(x_tfidf, y, groups),
            "structure_counts": eval_auc(x_struct, y, groups),
            "tfidf_plus_structure": eval_auc(
                hstack([x_tfidf, csr_matrix(x_struct)]), y, groups
            ),
            "haystack_sessions_only": eval_auc(x_struct[:, [3]], y, groups),
            "privileged_hops_answer_sessions": eval_auc(x_priv, y, groups),
        }

    return {
        "ar_vs_all_answerable": run(rows),
        "ar_vs_ie_only": run(
            [
                row
                for row in rows
                if row["qtype"] in {"answer_refusal", "information_extraction"}
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "${MOSAIC_DATA_ROOT}/"
            "human_filtered_evaluation_results/dataset/stage2_human_filtered"
        ),
    )
    parser.add_argument(
        "--interv-root",
        type=Path,
        default=Path(
            "${MOSAIC_DATA_ROOT}/"
            "evaluation_results/protocol_v1/modality_interventions"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/experiments/w5_mde_refusal_leak.json"),
    )
    args = parser.parse_args()
    args.data_dir = Path(_expand_data_path(str(args.data_dir)))
    args.interv_root = Path(_expand_data_path(str(args.interv_root)))

    contrasts = [
        (
            "ordered_minus_shuffled",
            "Ordered − shuffled (action_ordering)",
            "video_sparse_ordered_frames",
            "video_shuffled_frames",
            "action_ordering",
        ),
        (
            "event_minus_sparse_repetition",
            "Event − frames (repetition)",
            "video_ordered_event_count_description",
            "video_sparse_ordered_frames",
            "repetition_count",
        ),
        (
            "event_minus_sparse_transition",
            "Event − frames (transition)",
            "video_ordered_event_count_description",
            "video_sparse_ordered_frames",
            "state_transition",
        ),
    ]
    mde_rows = []
    for label, paper_row, left, right, challenge in contrasts:
        mde_rows.append(
            {
                "label": label,
                "paper_row": paper_row,
                "left": left,
                "right": right,
                "challenge": challenge,
                "intervention_unit": paired_mde(
                    args.interv_root, left, right, challenge, unit="intervention"
                ),
                "qid_unit": paired_mde(
                    args.interv_root, left, right, challenge, unit="qid"
                ),
            }
        )

    rows = load_rows(args.data_dir)
    refusal = refusal_block(rows)
    qid_mdes = [row["qid_unit"]["mde_points"] for row in mde_rows]
    payload = {
        "_description": (
            "W5 revised: QID-level video MDE; privileged negative-control vs "
            "observable refusal surface tests."
        ),
        "mde_method": (
            "MDE=(z_{1-α/2}+z_{1-β})*sd(delta)/sqrt(n) with α=0.05 two-sided, "
            "80% power, paired judge-score deltas. Primary unit = unique QID "
            "(average deltas when a QID has multiple interventions). Intervention "
            "unit reported for comparison with case-level tables."
        ),
        "video_mde": mde_rows,
        "video_mde_qid_summary_points": {
            "values": qid_mdes,
            "n_qids": [row["qid_unit"]["n"] for row in mde_rows],
            "min": min(qid_mdes),
            "max": max(qid_mdes),
        },
        "refusal_diagnostics": {
            "privileged_fields": [
                "n_required_hops",
                "n_answer_sessions",
                "question_type_label",
            ],
            "privileged_note": (
                "Construction/gold metadata. question_type=answer_refusal is a "
                "perfect label; hops/answer_sessions are zero after evidence strip. "
                "Never expose these in evaluation inputs."
            ),
            **refusal,
            "interpretation": (
                "Haystack length is matched (AUROC~0.52 vs all). Privileged "
                "construction fields achieve AUROC 1.0 by definition. Question "
                "TF-IDF remains partially predictive (AUROC~0.85 vs all; ~0.66 vs "
                "IE-only). Do not claim absence of all surface cues."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["video_mde_qid_summary_points"], indent=2))
    print(
        json.dumps(
            {
                "hay_all": refusal["ar_vs_all_answerable"]["haystack_sessions_only"],
                "tfidf_all": refusal["ar_vs_all_answerable"]["question_tfidf"],
                "tfidf_ie": refusal["ar_vs_ie_only"]["question_tfidf"],
                "priv": refusal["ar_vs_all_answerable"][
                    "privileged_hops_answer_sessions"
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
