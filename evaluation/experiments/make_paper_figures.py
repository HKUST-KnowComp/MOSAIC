"""Render the MOSAIC paper figures from aggregated ``build_paper_results`` output.

Every number is read from ``paper_results.json`` so the figures cannot drift
from the tables. Run after ``build_paper_results.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DEFAULT_RESULTS = (
    "${MOSAIC_DATA_ROOT}/evaluation_results"
    "/protocol_v1/paper_results/paper_results.json"
)
DEFAULT_OUTPUT = "latex/fig"


def _optional_data_path(value: str) -> Path | None:
    token = "${MOSAIC_DATA_ROOT}"
    if token not in value:
        return Path(value)
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        return None
    return Path(value.replace(token, root))


def _expand_data_path(value: str) -> str:
    token = "${MOSAIC_DATA_ROOT}"
    if token not in value:
        return value
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        raise SystemExit("Set MOSAIC_DATA_ROOT to the released data directory.")
    return value.replace(token, root)

# ACL two-column geometry. Heights are the uncompressed ones; the AAAI build
# used shorter figures only to fit a hard seven-page limit.
COL_W = 3.15
TEXT_W = 6.30

# Colourblind-safe (Okabe-Ito).
C_IMAGE = "#0072B2"
C_VIDEO = "#D55E00"
C_AUDIO = "#009E73"
C_STAGE = "#0072B2"
C_LOSS = "#D55E00"
C_GREY = "#666666"

STAGE_LABELS = {
    "gold_symbolic_facts": "Gold\nsymbolic",
    "gold_raw_evidence": "Gold\nnative",
    "retrieved_raw_all_delivered": "Retrieved,\nall delivered",
    "retrieved_raw_budgeted": "Retrieved,\ncapped",
    "end_to_end": "End-to-end\nRAG",
}
NATIVE_LADDER = list(STAGE_LABELS)

# (loss key, label) annotated between consecutive ladder rungs.
LADDER_LOSSES = [
    ("perception_cost", "perception"),
    ("retrieval_loss", "retrieval"),
    ("delivery_loss", "delivery"),
    ("production_query_effect", "query"),
]

INTERVENTION_ROWS = [
    ("image_raw_minus_ocr", "spatial_configuration",
     "Raw $-$ OCR-only (spatial config.)", C_IMAGE),
    ("image_raw_minus_ocr", "ocr_fine_detail",
     "Raw $-$ OCR-only (fine detail)", C_IMAGE),
    ("image_spatial_minus_object", "spatial_configuration",
     "Spatial desc. $-$ object-only (spatial config.)", C_IMAGE),
    ("video_order_benefit", "action_ordering",
     "Ordered $-$ shuffled frames (ordering)", C_VIDEO),
    ("video_event_description_minus_sparse", "repetition_count",
     "Event text $-$ sparse frames (repetition)", C_VIDEO),
    ("video_event_description_minus_sparse", "state_transition",
     "Event text $-$ sparse frames (transition)", C_VIDEO),
    ("audio_raw_minus_asr", "prosody",
     "Raw audio $-$ ASR (prosody)", C_AUDIO),
    ("audio_prosody_tag_recovery", "prosody",
     "ASR + gold prosody $-$ ASR (prosody)", C_AUDIO),
]

BUDGET_ORDER = [
    ("all_delivered", "All hits\ndelivered"),
    ("default_top20_media3", "Top-20,\n3 media"),
    ("top10_media3", "Top-10,\n3 media"),
    ("top5_media1", "Top-5,\n1 media"),
]

END_TO_END_POINTS = [
    ("rag_caption_text", "o", C_GREY),
    ("rag_omni_text", "o", C_GREY),
    ("react_caption_text", "s", C_GREY),
    ("react_omni_text", "s", C_GREY),
    ("amem_caption_text", "^", C_AUDIO),
    ("amem_omni_text", "^", C_AUDIO),
    ("memoryos_caption_text", "v", C_VIDEO),
    ("memoryos_omni_text", "v", C_VIDEO),
    ("rag_native_mm_mimo", "D", C_IMAGE),
    ("rag_native_mm_gemini_3_5", "D", C_IMAGE),
    ("react_native_mm_mimo", "*", C_IMAGE),
    ("react_native_mm_gemini_3_5", "*", C_IMAGE),
]

# Near-coincident systems share one label. Offsets are in points and were set
# by hand so no annotation overlaps a marker or another annotation.
CALIBRATION_LABELS = [
    (["memoryos_caption_text", "memoryos_omni_text"], "MemoryOS\n(text)",
     (12, 6), "left", "bottom"),
    (["rag_caption_text", "rag_omni_text"], "Naive RAG (text)",
     (12, 4), "left", "bottom"),
    (["react_caption_text", "react_omni_text"], "ReAct (text)",
     (13, -9), "left", "top"),
    (["amem_caption_text", "amem_omni_text"], "A-MEM (text)",
     (11, 9), "left", "bottom"),
    (["rag_native_mm_mimo"], "RAG native MiMo", (-11, -7), "right", "top"),
    (["rag_native_mm_gemini_3_5"], "RAG native Gemini", (11, 2),
     "left", "bottom"),
    (["react_native_mm_mimo"], "ReAct native MiMo", (11, -1),
     "left", "center"),
    (["react_native_mm_gemini_3_5"], "ReAct native Gemini", (-11, -2),
     "right", "top"),
]


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
    })


def _run_score(results: Mapping[str, Any], group: str, run: str) -> float:
    return float(results["run_metrics"][group][run]["mean_judge_score"])


def _refusal(results: Mapping[str, Any], run: str, field: str) -> float:
    return float(results["run_metrics"]["end_to_end"][run][field])


def figure_stage_ladder(results: Mapping[str, Any], out: Path) -> None:
    """Native pipeline cascade plus the write-time representation branch."""
    funnel = results["stage_failure_funnel"]
    scores = funnel["condition_scores"]
    losses = funnel["diagnostic_losses"]

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(TEXT_W, 2.45), gridspec_kw={"width_ratios": [2.05, 1.0]}
    )

    # --- (a) native pipeline cascade -----------------------------------
    ys = [scores[k] for k in NATIVE_LADDER]
    xs = list(range(len(ys)))
    gold_sym = scores["gold_symbolic_facts"]
    ax.axhspan(gold_sym, 1.0, color=C_LOSS, alpha=0.10, lw=0)
    ax.axhline(1.0, color=C_GREY, lw=0.6, ls=":")
    # Bottom-of-band, slightly above the gold-symbolic line and near the
    # horizontal center of panel (a) — clear of both the left spine and panel (b).
    ax.text(
        2.0, (gold_sym + 1.0) / 2,
        f"residual read  {funnel['residual_read_reasoning_error'] * 100:.1f} pts",
        ha="center", va="center", fontsize=7, color=C_LOSS,
    )
    ax.plot(xs, ys, "-o", color=C_STAGE, ms=4.5, zorder=3)

    # Drop arrows sit at the midpoint between rungs so they never collide with
    # the score labels printed above each marker.
    for i, (key, label) in enumerate(LADDER_LOSSES):
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[i], ys[i + 1]
        xm = (x0 + x1) / 2
        ax.plot([x0, xm], [y0, y0], color=C_GREY, lw=0.5, ls="--", zorder=1)
        ax.plot([xm, x1], [y1, y1], color=C_GREY, lw=0.5, ls="--", zorder=1)
        ax.annotate(
            "", xy=(xm, y1), xytext=(xm, y0),
            arrowprops=dict(arrowstyle="-|>", color=C_LOSS, lw=0.9,
                            shrinkA=0, shrinkB=0),
        )
        # Loss labels sit below the lower guide line, clear of the score
        # labels which are printed above each marker.
        ax.text(
            xm + 0.05, y1 - 0.012,
            f"{label}\n$-${losses[key] * 100:.1f}",
            fontsize=6.4, color=C_LOSS, va="top", ha="left",
        )

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=7, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([STAGE_LABELS[k] for k in NATIVE_LADDER], fontsize=6.8)
    ax.set_ylim(0.55, 1.03)
    ax.set_ylabel("Mean judge score")
    ax.set_title("(a) Native pipeline: where the score is lost", loc="left")
    ax.grid(axis="y")

    # --- (b) write-time representation branch --------------------------
    branch = ["gold_raw_evidence", "gold_structured_text", "gold_generic_text"]
    blabels = ["Native\nmedia", "Structured\ntext", "Generic\ncaptions+ASR"]
    bvals = [scores[k] for k in branch]
    bars = ax2.bar(range(3), bvals, color=[C_STAGE, C_AUDIO, C_LOSS],
                   width=0.58, zorder=3)
    for rect, val in zip(bars, bvals):
        ax2.annotate(f"{val:.3f}", (rect.get_x() + rect.get_width() / 2, val),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=7, weight="bold")
    textualization = (scores["gold_raw_evidence"]
                      - scores["gold_generic_text"]) * 100
    # Span arrow placed clear of the bars, at the right edge of the panel.
    xa = 2.55
    ax2.annotate(
        "", xy=(xa, scores["gold_generic_text"]),
        xytext=(xa, scores["gold_raw_evidence"]),
        arrowprops=dict(arrowstyle="<|-|>", color="black", lw=0.8),
    )
    ax2.text(xa - 0.10,
             (scores["gold_raw_evidence"] + scores["gold_generic_text"]) / 2,
             f"textualization\n$-${textualization:.1f} pts", fontsize=6.4,
             ha="right", va="center")
    ax2.text(1.0, 0.815,
             f"structured writer\nrecovers only "
             f"{losses['avoidable_write_loss'] * 100:+.1f} pts (n.s.)",
             fontsize=6.3, ha="center", va="bottom", color=C_AUDIO)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(blabels, fontsize=6.8)
    ax2.set_xlim(-0.6, 3.0)
    ax2.set_ylim(0.55, 0.855)
    ax2.set_title("(b) Write-time representation", loc="left")
    ax2.grid(axis="y")

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_interventions(results: Mapping[str, Any], out: Path) -> None:
    """Forest plot of within-item paired effects on their governing strata."""
    comps = results["modality_intervention_comparisons"]
    rows = []
    for name, challenge, label, colour in INTERVENTION_ROWS:
        by_ch = comps[name].get("by_challenge") or {}
        if challenge not in by_ch:
            continue
        eff = by_ch[challenge]
        lo, hi = eff["item_bootstrap_95_ci"]
        rows.append((
            label, colour,
            float(eff["mean_paired_effect_left_minus_right"]),
            float(lo), float(hi), int(eff["n"]),
        ))

    # Optional selector-sensitivity arm (correct vs swapped gold prosody tag).
    swap_path = _optional_data_path(
        "${MOSAIC_DATA_ROOT}/"
        "evaluation_results/protocol_v1/modality_interventions/"
        "prosody_selector_sensitivity/paired_correct_minus_swapped.json"
    )
    if swap_path is not None and swap_path.is_file():
        swap = json.loads(swap_path.read_text(encoding="utf-8"))
        lo, hi = swap["ci95_qid_bootstrap"]
        rows.append((
            "Correct prosody tag $-$ swapped tag (prosody)",
            C_AUDIO,
            float(swap["mean_delta_qid"]),
            float(lo),
            float(hi),
            int(swap["num_unique_qids"]),
        ))

    fig, ax = plt.subplots(figsize=(COL_W, 2.55))
    ys = list(range(len(rows)))[::-1]
    for y, (label, colour, eff, lo, hi, n) in zip(ys, rows):
        significant = lo > 0 or hi < 0
        ax.plot([lo, hi], [y, y], color=colour, lw=1.4,
                alpha=1.0 if significant else 0.55, solid_capstyle="round")
        ax.plot([eff], [y], "o" if significant else "o",
                color=colour if significant else "white",
                mec=colour, mew=1.1, ms=5, zorder=3)
        ax.text(hi + 0.012, y, f"$n$={n}", fontsize=6, va="center",
                color=C_GREY)

    ax.axvline(0.0, color="black", lw=0.7, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    # Modality is encoded in the tick-label colour, which removes the need for
    # a legend that would otherwise sit on top of the near-null rows.
    for tick, (_, colour, *_rest) in zip(ax.get_yticklabels(), rows):
        tick.set_color(colour)
    ax.set_xlabel("Paired effect (95% CI)")
    ax.set_xlim(-0.13, 0.47)
    ax.grid(axis="x")
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_budget(results: Mapping[str, Any], out: Path) -> None:
    """Delivery budget curve: score and complete-chain delivery together."""
    curve = results["delivery_budget_curve"]
    points = curve["budget_points"]
    all_del = curve["all_delivered"]

    scores, chains, labels = [], [], []
    for key, label in BUDGET_ORDER:
        labels.append(label)
        if key == "all_delivered":
            scores.append(
                float(points["default_top20_media3"]["delivery_loss"]["left_mean"])
            )
            chains.append(float(all_del["complete_chain_delivery_rate"]))
        else:
            scores.append(
                float(points[key]["delivery_loss"]["right_mean"])
            )
            chains.append(float(points[key]["complete_chain_delivery_rate"]))

    fig, ax = plt.subplots(figsize=(COL_W, 2.35))
    xs = list(range(len(labels)))
    # The two series cross at the left edge, so the first point's labels are
    # nudged sideways instead of straight up/down.
    score_off = [(-16, 2), (0, 7), (0, 7), (0, 7)]
    chain_off = [(14, -4), (0, -12), (0, -12), (0, -12)]
    ax.plot(xs, scores, "-o", color=C_STAGE, ms=4.5, label="Judge score")
    for x, y, off in zip(xs, scores, score_off):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=off, ha="center", fontsize=6.5, color=C_STAGE)
    ax.set_ylabel("Mean judge score", color=C_STAGE)
    ax.tick_params(axis="y", colors=C_STAGE)
    ax.set_ylim(0.45, 0.80)

    ax3 = ax.twinx()
    ax3.spines["right"].set_visible(True)
    ax3.plot(xs, chains, "--s", color=C_LOSS, ms=4, label="Complete-chain delivery")
    for x, y, off in zip(xs, chains, chain_off):
        ax3.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                     xytext=off, ha="center", fontsize=6.5, color=C_LOSS)
    ax3.set_ylabel("Complete-chain delivery", color=C_LOSS)
    ax3.tick_params(axis="y", colors=C_LOSS)
    ax3.set_ylim(0.10, 0.78)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.6)
    ax.grid(axis="y")
    ax.text(0.02, 0.04,
            "identical ranked manifest;\nrequired-fact R@50 $=$ "
            f"{all_del['fact_recall_at_k']['50']:.3f} throughout",
            transform=ax.transAxes, fontsize=6.2, color=C_GREY, va="bottom")
    lines = ax.get_lines()[:1] + ax3.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="upper right",
              frameon=False, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_calibration(results: Mapping[str, Any], out: Path) -> None:
    """Answering vs abstention, with false refusal encoded as marker size."""
    metrics_all = results["run_metrics"]["end_to_end"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.60))
    ax.axhspan(0.80, 1.0, color=C_VIDEO, alpha=0.07, lw=0)

    coords: dict[str, tuple[float, float]] = {}
    for run, marker, colour in END_TO_END_POINTS:
        metrics = metrics_all.get(run)
        if not metrics:
            continue
        x = float(metrics["mean_judge_score"])
        y = float(metrics["correct_refusal_rate"])
        false_ref = float(metrics["false_refusal_rate"])
        coords[run] = (x, y)
        ax.scatter(x, y, s=20 + 300 * false_ref, marker=marker, c=colour,
                   alpha=0.5, edgecolors="black", linewidths=0.5, zorder=3)

    for runs, label, offset, ha, va in CALIBRATION_LABELS:
        pts = [coords[r] for r in runs if r in coords]
        if not pts:
            continue
        ax.annotate(
            label,
            (sum(p[0] for p in pts) / len(pts),
             sum(p[1] for p in pts) / len(pts)),
            textcoords="offset points", xytext=offset, ha=ha, va=va,
            fontsize=5.9, color="black", zorder=4,
        )

    ax.set_xlabel("Answerable judge score")
    ax.set_ylabel("Correct refusal rate")
    ax.set_xlim(0.17, 0.86)
    ax.set_ylim(0.33, 1.0)
    ax.grid(True)
    ax.text(0.015, 0.035, "marker area $\\propto$ false refusal",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.2, color=C_GREY)
    ax.text(0.985, 0.955,
            "unoccupied: no system both\nanswers and abstains well",
            transform=ax.transAxes, fontsize=6.2, color=C_GREY,
            ha="right", va="top")
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


FAMILY_COLOURS = {
    "naive_rag": C_IMAGE,
    "react": C_AUDIO,
    "amem": "#CC79A7",
    "memoryos": "#8064A2",
}

QTYPE_LABELS = {
    "information_extraction": "IE",
    "msr_selection": "MSR-S",
    "msr_bridge": "MSR-B",
    "knowledge_update": "KU",
    "temporal_reasoning": "TR",
}


def _family(run: str) -> str:
    for key in ("amem", "memoryos", "react"):
        if run.startswith(key):
            return key
    return "naive_rag"


def _ranked_end_to_end(results: Mapping[str, Any]) -> list[tuple[str, str, float]]:
    metrics_all = results["run_metrics"]["end_to_end"]
    rows = [
        (run, str(metrics.get("label", run)), float(metrics["mean_judge_score"]))
        for run, metrics in metrics_all.items()
    ]
    return sorted(rows, key=lambda row: row[2])


def figure_overall_score(results: Mapping[str, Any], out: Path) -> None:
    """Track B systems ranked by answerable judge score, with clustered CIs."""
    metrics_all = results["run_metrics"]["end_to_end"]
    rows = _ranked_end_to_end(results)
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.82, 0.32 * len(rows) + 0.9))

    ys = list(range(len(rows)))
    for y, (run, label, score) in zip(ys, rows):
        colour = FAMILY_COLOURS[_family(run)]
        lo, hi = metrics_all[run]["persona_clustered_bootstrap_95_ci"]
        ax.barh(y, score, color=colour, alpha=0.85, height=0.62)
        ax.errorbar(score, y, xerr=[[score - lo], [hi - score]], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=2.0, zorder=4)
        ax.text(hi + 0.012, y, f"{score:.3f}", va="center", fontsize=7.0)

    ax.set_yticks(ys)
    ax.set_yticklabels([label for _, label, _ in rows], fontsize=7.4)
    ax.set_xlabel("Mean judge score (784 answerable items)")
    ax.set_xlim(0, 0.88)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    handles = [
        Line2D([], [], color=colour, lw=5, label=name)
        for name, colour in (
            ("Naive RAG", FAMILY_COLOURS["naive_rag"]),
            ("ReAct", FAMILY_COLOURS["react"]),
            ("A-MEM", FAMILY_COLOURS["amem"]),
            ("MemoryOS", FAMILY_COLOURS["memoryos"]),
        )
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7,
              ncol=2, handlelength=1.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def _qtype_matrix(
    results: Mapping[str, Any],
) -> tuple[list[str], list[str], list[list[float]]]:
    metrics_all = results["run_metrics"]["end_to_end"]
    breakdowns = results["breakdowns"]["end_to_end"]
    rows = _ranked_end_to_end(results)[::-1]
    cols = list(QTYPE_LABELS) + ["answer_refusal"]
    matrix: list[list[float]] = []
    for run, _, _ in rows:
        by_type = breakdowns.get(run, {}).get("question_type", {})
        line = [
            float(by_type.get(key, {}).get("mean_judge_score", float("nan")))
            for key in QTYPE_LABELS
        ]
        line.append(float(metrics_all[run]["correct_refusal_rate"]))
        matrix.append(line)
    labels = [label for _, label, _ in rows]
    headers = [QTYPE_LABELS[key] for key in QTYPE_LABELS] + ["AR"]
    return labels, headers, matrix


def figure_heatmap(results: Mapping[str, Any], out: Path) -> None:
    """System x question-type score matrix."""
    labels, headers, matrix = _qtype_matrix(results)
    fig, ax = plt.subplots(figsize=(TEXT_W * 0.80, 0.30 * len(labels) + 1.1))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(headers)))
    ax.set_xticklabels(headers, fontsize=7.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.4)
    ax.set_xlabel("Question type (AR $=$ correct-refusal rate)")
    for i, line in enumerate(matrix):
        for j, value in enumerate(line):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.6,
                    color="black")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    bar.ax.tick_params(labelsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure_by_question_type(results: Mapping[str, Any], out: Path) -> None:
    """One panel per question type, systems ranked within each panel."""
    labels, headers, matrix = _qtype_matrix(results)
    colours = [FAMILY_COLOURS[_family(run)]
               for run, _, _ in _ranked_end_to_end(results)[::-1]]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 4.3), sharex=True)

    for idx, header in enumerate(headers):
        ax = axes.flat[idx]
        column = [(labels[i], colours[i], matrix[i][idx])
                  for i in range(len(labels))]
        column.sort(key=lambda row: row[2])
        ys = list(range(len(column)))
        ax.barh(ys, [row[2] for row in column],
                color=[row[1] for row in column], alpha=0.85, height=0.66)
        ax.set_yticks(ys)
        ax.set_yticklabels([row[0] for row in column], fontsize=5.6)
        ax.set_title(header, fontsize=8)
        ax.set_xlim(0, 1.0)
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=6.5)

    axes.flat[-1].set_visible(len(headers) >= len(axes.flat))
    fig.supxlabel("Mean judge score", fontsize=8)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.results = _expand_data_path(str(args.results))

    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    targets = [
        ("stage_ladder.pdf", figure_stage_ladder),
        ("modality_forest.pdf", figure_interventions),
        ("delivery_budget.pdf", figure_budget),
        ("calibration.pdf", figure_calibration),
        ("stat/overall_score.png", figure_overall_score),
        ("stat/heatmap.png", figure_heatmap),
        ("stat/by_question_type.png", figure_by_question_type),
    ]
    for name, builder in targets:
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        builder(results, path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
