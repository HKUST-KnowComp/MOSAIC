#!/usr/bin/env bash
# JUDGE-PANEL-E1: secondary GLM-5.2 + Grok-4.5 judges + agreement stats.
#
# Runs only for Track B paper rows whose primary judge/ output already exists
# (details_judged.jsonl + summary_judged.json). Skips runs that already have
# a complete panel unless --force is passed.
#
# Usage:
#   bash evaluation/judge/run_judge_panel_e1.sh
#   bash evaluation/judge/run_judge_panel_e1.sh --dry-run
#   bash evaluation/judge/run_judge_panel_e1.sh --force
#   EVAL_OUTPUT_ROOT=/path/to/protocol_v1 bash evaluation/judge/run_judge_panel_e1.sh
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-mm-atlas}"
if [[ -z "${EVAL_OUTPUT_ROOT:-}" && -z "${MOSAIC_DATA_ROOT:-}" ]]; then
  echo "Set MOSAIC_DATA_ROOT or EVAL_OUTPUT_ROOT." >&2
  exit 1
fi
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1}"
DRY_RUN=0
FORCE=0
CONTINUE_ON_ERROR=0

# Track B paper rows only (parity gates excluded). Same list as EXPERIMENT_RUNBOOK.md.
TRACK_B_RUNS=(
  caption_text/rag
  omni_text/rag
  omni_mm_xiaomi/rag
  omni_mm_gemini_3_5/rag
  caption_text/react_adaptive
  omni_text/react_adaptive
  omni_mm_xiaomi/react_adaptive
  omni_mm_gemini_3_5/react_adaptive
  caption_text/amem_native_end_to_end
  omni_text/amem_native_end_to_end
  caption_text/memoryos_native_end_to_end
  omni_text/memoryos_native_end_to_end
)

usage() {
  cat <<EOF
Usage: bash evaluation/judge/run_judge_panel_e1.sh [options]

Options:
  --output-root PATH   Protocol output root (default: \$EVAL_OUTPUT_ROOT or protocol_v1)
  --dry-run            Print planned actions without calling the judge
  --force              Re-run GLM/Grok/agreement even when outputs already exist
  --continue-on-error  Keep going after a failed run instead of exiting
  -h, --help           Show this help
EOF
}

while (($#)); do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="${2:?missing value for --output-root}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run_cmd() {
  if ((DRY_RUN)); then
    printf '  [dry-run] %s\n' "$*"
    return 0
  fi
  "$@"
}

primary_judge_done() {
  local run_dir="$1"
  [[ -f "$run_dir/judge/details_judged.jsonl" && -f "$run_dir/judge/summary_judged.json" ]]
}

panel_done() {
  local run_dir="$1"
  [[ -f "$run_dir/judge_glm_5_2/summary_judged.json" \
     && -f "$run_dir/judge_grok_4_5/summary_judged.json" \
     && -f "$run_dir/judge_agreement.json" ]]
}

echo "Repository:  $REPO_ROOT"
echo "Conda env:   $CONDA_ENV"
echo "Output root: $OUTPUT_ROOT"
echo "Mode:        $([ "$DRY_RUN" -eq 1 ] && echo dry-run || echo execute)$([ "$FORCE" -eq 1 ] && echo ' (force)' || true)"
echo

ready=()
skipped_no_primary=()
skipped_complete=()

for run in "${TRACK_B_RUNS[@]}"; do
  run_dir="$OUTPUT_ROOT/$run"
  if ! primary_judge_done "$run_dir"; then
    skipped_no_primary+=("$run")
    continue
  fi
  if ((FORCE == 0)) && panel_done "$run_dir"; then
    skipped_complete+=("$run")
    continue
  fi
  ready+=("$run")
done

echo "Primary judge done, panel pending: ${#ready[@]}"
for run in "${ready[@]+"${ready[@]}"}"; do
  echo "  - $run"
done
echo
echo "Skip (primary judge missing): ${#skipped_no_primary[@]}"
for run in "${skipped_no_primary[@]+"${skipped_no_primary[@]}"}"; do
  echo "  - $run"
done
echo
echo "Skip (panel already complete): ${#skipped_complete[@]}"
for run in "${skipped_complete[@]+"${skipped_complete[@]}"}"; do
  echo "  - $run"
done
echo

if ((${#ready[@]} == 0)); then
  echo "Nothing to run."
  exit 0
fi

failed=()
for run in "${ready[@]}"; do
  run_dir="$OUTPUT_ROOT/$run"
  details="$run_dir/details.jsonl"
  echo "========================================================================"
  echo "PANEL: $run"
  echo "========================================================================"

  if [[ ! -f "$details" ]]; then
    echo "ERROR: missing $details" >&2
    failed+=("$run")
    ((CONTINUE_ON_ERROR)) || exit 1
    continue
  fi

  if ((FORCE)) || [[ ! -f "$run_dir/judge_glm_5_2/summary_judged.json" ]]; then
    run_cmd conda run --no-capture-output -n "$CONDA_ENV" \
      python evaluation/judge/eval_judge.py \
      --config evaluation/judge/config_judge_glm_5_2.json \
      --input "$details" \
      --output-dir "$run_dir/judge_glm_5_2" \
      || { failed+=("$run:glm"); ((CONTINUE_ON_ERROR)) || exit 1; continue; }
  else
    echo "  skip GLM (exists)"
  fi

  if ((FORCE)) || [[ ! -f "$run_dir/judge_grok_4_5/summary_judged.json" ]]; then
    run_cmd conda run --no-capture-output -n "$CONDA_ENV" \
      python evaluation/judge/eval_judge.py \
      --config evaluation/judge/config_judge_grok_4_5.json \
      --input "$details" \
      --output-dir "$run_dir/judge_grok_4_5" \
      || { failed+=("$run:grok"); ((CONTINUE_ON_ERROR)) || exit 1; continue; }
  else
    echo "  skip Grok (exists)"
  fi

  if ((FORCE)) || [[ ! -f "$run_dir/judge_agreement.json" ]]; then
    run_cmd conda run --no-capture-output -n "$CONDA_ENV" \
      python evaluation/judge/judge_agreement.py \
      "$run_dir/judge" \
      "$run_dir/judge_glm_5_2" \
      "$run_dir/judge_grok_4_5" \
      --output "$run_dir/judge_agreement.json" \
      || { failed+=("$run:agreement"); ((CONTINUE_ON_ERROR)) || exit 1; continue; }
  else
    echo "  skip agreement (exists)"
  fi
done

echo
echo "========================================================================"
echo "JUDGE-PANEL-E1 summary"
echo "========================================================================"
echo "Processed: ${#ready[@]}"
echo "Failed:    ${#failed[@]}"
for item in "${failed[@]+"${failed[@]}"}"; do
  echo "  - $item"
done

complete=0
for run in "${TRACK_B_RUNS[@]}"; do
  if panel_done "$OUTPUT_ROOT/$run"; then
    complete=$((complete + 1))
  fi
done
echo "Panel complete: $complete / ${#TRACK_B_RUNS[@]}"
((${#failed[@]} == 0))
