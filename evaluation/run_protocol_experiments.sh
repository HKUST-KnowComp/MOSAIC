#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# RAG / ReAct need the embedding stack in omni-emb (sentence-transformers + Qwen).
# Override with CONDA_ENV=mm-atlas only for non-embedding protocol helpers if needed.
CONDA_ENV="${CONDA_ENV:-omni-emb}"
AMEM_CONDA_ENV="${AMEM_CONDA_ENV:-a-mem}"
MEMORYOS_CONDA_ENV="${MEMORYOS_CONDA_ENV:-memory_os}"
# Protocol checks / py_compile stay on mm-atlas (has the controlled-experiment deps).
CHECKS_CONDA_ENV="${CHECKS_CONDA_ENV:-mm-atlas}"
if [[ -z "${EVAL_OUTPUT_ROOT:-}" && -z "${MOSAIC_DATA_ROOT:-}" ]]; then
  echo "Set MOSAIC_DATA_ROOT or EVAL_OUTPUT_ROOT." >&2
  exit 1
fi
OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1}"
DEFAULT_VARIANTS="caption_text,omni_text,omni_mm_xiaomi,omni_mm_gemini,omni_mm_gemini_3_5"

# Load repo .env + provider aliases into this shell so require_api_key sees them.
# Python evaluators also load .env themselves via evaluation/env_bootstrap.py.
if [[ -f "$REPO_ROOT/.env" ]]; then
  eval "$(
    conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
from __future__ import annotations
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path("evaluation").resolve()))
from env_bootstrap import ensure_env_loaded

ensure_env_loaded()
for key in (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_URL",
    "DEEPSEEK_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "YUNWU_API_KEY",
    "YUNWU_BASE_URL",
    "GEMINI_KEY",
    "OPENAI_API_KEY",
):
    value = os.environ.get(key)
    if value:
        print(f"export {key}={shlex.quote(value)}")
PY
  )"
fi

SUITE=""
VARIANTS="${EVAL_VARIANTS:-$DEFAULT_VARIANTS}"
DRY_RUN=0
RESUME=0
REUSE_BASELINE=0
MAX_PERSONAS=""
PERSONA_FILTER=""
QUERY_WORKERS=""

usage() {
  cat <<'EOF'
Usage: bash evaluation/run_protocol_experiments.sh --suite SUITE [options]

Suites:
  checks     Dependency and deterministic protocol checks; no API calls
  smoke      One-persona run across all supported backends
  rag        Full Naive RAG baselines
  parity     Naive RAG plus one-search ReAct parity and strict comparison
  adaptive   Full repaired adaptive ReAct runs
  amem       Full A-MEM runs (caption_text and omni_text only)
  memoryos   Full MemoryOS runs (caption_text and omni_text only)
  memory     Full A-MEM and MemoryOS runs (text variants only)
  all        Checks, parity, adaptive ReAct, A-MEM, and MemoryOS

Options:
  --variants CSV          caption_text, omni_text, omni_mm_xiaomi,
                          omni_mm_gemini, omni_mm_gemini_3_5
  --output-root PATH      Output root (or set EVAL_OUTPUT_ROOT)
  --max-personas N        Limit every evaluator run
  --persona-filter CSV    Limit to matching persona UUID substrings
  --query-workers N       Override evaluator worker counts
  --resume                Retry RAG/ReAct terminal failures in existing output
  --reuse-baseline        In parity suite, use an existing RAG output
  --dry-run               Print commands without running them
  -h, --help              Show this help

Examples:
  bash evaluation/run_protocol_experiments.sh --suite smoke --variants caption_text
  bash evaluation/run_protocol_experiments.sh --suite memory --variants caption_text,omni_text
  bash evaluation/run_protocol_experiments.sh --suite parity --dry-run
  EVAL_OUTPUT_ROOT=/data/.../protocol_v1 bash evaluation/run_protocol_experiments.sh --suite all
EOF
}

while (($#)); do
  case "$1" in
    --suite)
      SUITE="${2:?missing value for --suite}"
      shift 2
      ;;
    --variants)
      VARIANTS="${2:?missing value for --variants}"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="${2:?missing value for --output-root}"
      shift 2
      ;;
    --max-personas)
      MAX_PERSONAS="${2:?missing value for --max-personas}"
      shift 2
      ;;
    --persona-filter)
      PERSONA_FILTER="${2:?missing value for --persona-filter}"
      shift 2
      ;;
    --query-workers)
      QUERY_WORKERS="${2:?missing value for --query-workers}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --reuse-baseline)
      REUSE_BASELINE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SUITE" ]]; then
  echo "--suite is required" >&2
  usage >&2
  exit 2
fi
case "$SUITE" in
  checks|smoke|rag|parity|adaptive|amem|memoryos|memory|all) ;;
  *)
    echo "Unknown suite: $SUITE" >&2
    exit 2
    ;;
esac

declare -A RAG_SCRIPT=(
  [caption_text]="evaluation/rag/eval_rag_text.py"
  [omni_text]="evaluation/rag/eval_rag_text.py"
  [omni_mm_xiaomi]="evaluation/rag/eval_rag_mm.py"
  [omni_mm_gemini]="evaluation/rag/eval_rag_mm.py"
  [omni_mm_gemini_3_5]="evaluation/rag/eval_rag_mm.py"
)
declare -A RAG_CONFIG=(
  [caption_text]="evaluation/rag/config_eval_caption_text_deepseekv4flash.json"
  [omni_text]="evaluation/rag/config_eval_omni_text_deepseekv4flash.json"
  [omni_mm_xiaomi]="evaluation/rag/config_eval_omni_mm_xiaomi.json"
  [omni_mm_gemini]="evaluation/rag/config_eval_omni_mm_gemini.json"
  [omni_mm_gemini_3_5]="evaluation/rag/config_eval_omni_mm_gemini_3_5.json"
)
declare -A REACT_CONFIG=(
  [caption_text]="evaluation/react/config_eval_react_caption_text.json"
  [omni_text]="evaluation/react/config_eval_react_omni_text.json"
  [omni_mm_xiaomi]="evaluation/react/config_eval_react_omni_mm.json"
  [omni_mm_gemini]="evaluation/react/config_eval_react_omni_mm_gemini.json"
  [omni_mm_gemini_3_5]="evaluation/react/config_eval_react_omni_mm_gemini_3_5.json"
)
declare -A REQUIRED_API_ENV=(
  [caption_text]="DEEPSEEK_API_KEY"
  [omni_text]="DEEPSEEK_API_KEY"
  [omni_mm_xiaomi]="OPENROUTER_API_KEY"
  [omni_mm_gemini]="YUNWU_API_KEY"
  [omni_mm_gemini_3_5]="YUNWU_API_KEY"
)
declare -A AMEM_CONFIG=(
  [caption_text]="evaluation/amem/query_caption_config.json"
  [omni_text]="evaluation/amem/query_omni_config.json"
)
declare -A MEMORYOS_CONFIG=(
  [caption_text]="evaluation/memoryos/memoryos_eval_config_caption.json"
  [omni_text]="evaluation/memoryos/memoryos_eval_config_omni.json"
)

IFS=',' read -r -a SELECTED_VARIANTS <<< "$VARIANTS"
for variant in "${SELECTED_VARIANTS[@]}"; do
  if [[ -z "${RAG_CONFIG[$variant]+x}" ]]; then
    echo "Unknown variant: $variant" >&2
    exit 2
  fi
done

run_cmd() {
  printf ' +'
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN == 0)); then
    "$@"
  fi
}

run_python() {
  run_env_python "$CONDA_ENV" "$@"
}

run_env_python() {
  local env_name="$1"
  shift
  run_cmd conda run --no-capture-output -n "$env_name" python "$@"
}

require_api_key() {
  local variant="$1"
  local variable="${REQUIRED_API_ENV[$variant]}"
  if ((DRY_RUN == 0)) && [[ -z "${!variable:-}" ]]; then
    echo "Required environment variable is unset: $variable ($variant)" >&2
    exit 1
  fi
}

common_args() {
  COMMON_ARGS=()
  if [[ -n "$MAX_PERSONAS" ]]; then
    COMMON_ARGS+=(--max-personas "$MAX_PERSONAS")
  fi
  if [[ -n "$PERSONA_FILTER" ]]; then
    COMMON_ARGS+=(--persona-filter "$PERSONA_FILTER")
  fi
  if [[ -n "$QUERY_WORKERS" ]]; then
    COMMON_ARGS+=(--query-workers "$QUERY_WORKERS")
  fi
  if ((RESUME)); then
    COMMON_ARGS+=(--resume)
  fi
}

run_checks() {
  echo "== Protocol checks =="
  run_env_python "$CHECKS_CONDA_ENV" -m py_compile \
    evaluation/_common.py \
    evaluation/protocol.py \
    evaluation/experiments/answer_harness.py \
    evaluation/experiments/run_stage_decomposition.py \
    evaluation/experiments/run_modality_interventions.py \
    evaluation/experiments/run_fixed_reader_comparison.py \
    evaluation/experiments/build_paper_results.py \
    evaluation/rag/eval_rag_text.py \
    evaluation/rag/eval_rag_mm.py \
    evaluation/react/eval_rag_react.py \
    evaluation/amem/query_index.py \
    evaluation/memoryos/eval_questions.py
  run_env_python "$CHECKS_CONDA_ENV" evaluation/run_sanity_checks.py --json
  run_env_python "$AMEM_CONDA_ENV" -m py_compile \
    evaluation/protocol.py \
    evaluation/amem/_common.py \
    evaluation/amem/query_index.py
  run_env_python "$AMEM_CONDA_ENV" -c \
    "import sys; sys.path.insert(0, 'evaluation/amem'); import _common; print('A-MEM dependencies: ok')"
  run_env_python "$MEMORYOS_CONDA_ENV" -m py_compile \
    evaluation/protocol.py \
    evaluation/memoryos/common.py \
    evaluation/memoryos/eval_questions.py
  run_env_python "$MEMORYOS_CONDA_ENV" -c \
    "import sys; sys.path.insert(0, 'evaluation/memoryos'); import common; print('MemoryOS dependencies: ok')"
  if ((DRY_RUN)); then
    run_env_python "$CHECKS_CONDA_ENV" evaluation/audit_session_collisions.py \
      --output "$OUTPUT_ROOT/session_collision_audit.json" --quiet
  else
    mkdir -p "$OUTPUT_ROOT"
    run_env_python "$CHECKS_CONDA_ENV" evaluation/audit_session_collisions.py \
      --output "$OUTPUT_ROOT/session_collision_audit.json" --quiet
  fi
}

text_memory_args() {
  TEXT_MEMORY_ARGS=()
  if [[ -n "$MAX_PERSONAS" ]]; then
    TEXT_MEMORY_ARGS+=(--max-personas "$MAX_PERSONAS")
  fi
  if [[ -n "$PERSONA_FILTER" ]]; then
    TEXT_MEMORY_ARGS+=(--persona-filter "$PERSONA_FILTER")
  fi
}

run_amem() {
  local variant="$1"
  local config="${AMEM_CONFIG[$variant]:-}"
  if [[ -z "$config" ]]; then
    echo "Skipping A-MEM unsupported variant: $variant"
    return
  fi
  require_api_key "$variant"
  text_memory_args
  if [[ -n "$QUERY_WORKERS" ]]; then
    TEXT_MEMORY_ARGS+=(--query-workers "$QUERY_WORKERS")
  fi
  run_env_python "$AMEM_CONDA_ENV" evaluation/amem/query_index.py \
    --config "$config" \
    --output-dir "$OUTPUT_ROOT/$variant/amem_native_end_to_end" \
    "${TEXT_MEMORY_ARGS[@]}"
}

run_memoryos() {
  local variant="$1"
  local config="${MEMORYOS_CONFIG[$variant]:-}"
  if [[ -z "$config" ]]; then
    echo "Skipping MemoryOS unsupported variant: $variant"
    return
  fi
  require_api_key "$variant"
  text_memory_args
  if [[ -n "$QUERY_WORKERS" ]]; then
    TEXT_MEMORY_ARGS+=(--eval-workers "$QUERY_WORKERS")
  fi
  run_env_python "$MEMORYOS_CONDA_ENV" evaluation/memoryos/eval_questions.py \
    --config "$config" \
    --output-dir "$OUTPUT_ROOT/$variant/memoryos_native_end_to_end" \
    "${TEXT_MEMORY_ARGS[@]}"
}

run_rag() {
  local variant="$1"
  local output_dir="$OUTPUT_ROOT/$variant/rag"
  require_api_key "$variant"
  common_args
  run_python "${RAG_SCRIPT[$variant]}" \
    --config "${RAG_CONFIG[$variant]}" \
    --output-dir "$output_dir" \
    "${COMMON_ARGS[@]}"
}

run_parity() {
  local variant="$1"
  local baseline_dir="$OUTPUT_ROOT/$variant/rag"
  local parity_dir="$OUTPUT_ROOT/$variant/react_one_search_parity"
  require_api_key "$variant"
  if ((REUSE_BASELINE == 0)); then
    run_rag "$variant"
  elif ((DRY_RUN == 0)) && [[ ! -f "$baseline_dir/details.jsonl" ]]; then
    echo "Missing reusable baseline: $baseline_dir/details.jsonl" >&2
    exit 1
  fi
  common_args
  run_python evaluation/react/eval_rag_react.py \
    --config "${REACT_CONFIG[$variant]}" \
    --react-mode one_search_parity \
    --output-dir "$parity_dir" \
    "${COMMON_ARGS[@]}"
  # Strict prediction-equality comparison is informational only: hosted
  # reasoners are not bit-deterministic at temperature 0, so answer strings
  # can differ on identical delivered evidence. Its divergence rate is
  # recorded for the robustness section.
  run_python evaluation/compare_parity_runs.py \
    "$baseline_dir" "$parity_dir" \
    --output "$parity_dir/parity_comparison_strict.json" \
    || echo "NOTE: strict answer-string divergence is expected with hosted reasoners (recorded in parity_comparison_strict.json); the binding gate follows."
  # The binding parity gate: query, retrieval, and delivery must be identical.
  run_python evaluation/compare_parity_runs.py \
    "$baseline_dir" "$parity_dir" \
    --ignore-predictions \
    --output "$parity_dir/parity_comparison.json"
}

run_adaptive() {
  local variant="$1"
  local output_dir="$OUTPUT_ROOT/$variant/react_adaptive"
  require_api_key "$variant"
  common_args
  run_python evaluation/react/eval_rag_react.py \
    --config "${REACT_CONFIG[$variant]}" \
    --react-mode adaptive_react \
    --output-dir "$output_dir" \
    "${COMMON_ARGS[@]}"
}

echo "Repository:  $REPO_ROOT"
echo "Conda env:   $CONDA_ENV"
echo "Checks env:  $CHECKS_CONDA_ENV"
echo "A-MEM env:   $AMEM_CONDA_ENV"
echo "MemoryOS env: $MEMORYOS_CONDA_ENV"
echo "Output root: $OUTPUT_ROOT"
echo "Suite:       $SUITE"
echo "Variants:    ${SELECTED_VARIANTS[*]}"

if [[ "$SUITE" == "smoke" ]]; then
  MAX_PERSONAS="${MAX_PERSONAS:-1}"
  QUERY_WORKERS="${QUERY_WORKERS:-1}"
fi

case "$SUITE" in
  checks)
    run_checks
    ;;
  smoke)
    run_checks
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_parity "$variant"
      run_adaptive "$variant"
      run_amem "$variant"
      run_memoryos "$variant"
    done
    ;;
  rag)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_rag "$variant"
    done
    ;;
  parity)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_parity "$variant"
    done
    ;;
  adaptive)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_adaptive "$variant"
    done
    ;;
  amem)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_amem "$variant"
    done
    ;;
  memoryos)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_memoryos "$variant"
    done
    ;;
  memory)
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_amem "$variant"
      run_memoryos "$variant"
    done
    ;;
  all)
    run_checks
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_parity "$variant"
    done
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_adaptive "$variant"
    done
    for variant in "${SELECTED_VARIANTS[@]}"; do
      run_amem "$variant"
      run_memoryos "$variant"
    done
    ;;
esac
