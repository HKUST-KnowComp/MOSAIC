# Evaluation Commands

All commands run from repo root: `cd /path/to/MOSAIC`

The complete one-command-per-run checklist, including Experiments 2--4 and
every matching judge command, is in
[EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md). Controlled experiment inference
(E2–E4) runs in `omni-emb` (Jina embedding stack / `peft`). Judges and paper
aggregation stay in `mm-atlas`; A-MEM / MemoryOS source exports use `a-mem` /
`memory_os`.

## Controlled Paper Experiments

These are the full controlled inference commands. They use the configured
result roots. The matching judge command and exact result path for every row
are listed in [EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md).

### Experiment 2

Prepare the one frozen retrieval manifest before the three retrieved
conditions. This requires the complete production rewrite manifest at
`evaluation/manifests/fixed_rewrite_queries.json`; the artifact freezes both
the original-query diagnostic ranking and the fixed-rewrite production ranking:

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --prepare-retrieval-manifest
```

- [ ] `gold_symbolic_facts`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_symbolic_facts
```

- [ ] `gold_raw_evidence`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_raw_evidence
```

- [ ] `gold_generic_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_generic_text
```

- [ ] `gold_structured_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_structured_text
```

- [ ] `retrieved_raw_all_delivered`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_all_delivered
```

- [ ] `retrieved_raw_budgeted`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted
```

- [ ] `end_to_end`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition end_to_end
```

`end_to_end` uses the frozen production fixed-rewrite ranking. The two
`retrieved_raw_*` conditions use the frozen original-query ranking.

#### Delivery budget sweep

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted --config evaluation/configs/stage_decomposition_budget_top10.json

conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted --config evaluation/configs/stage_decomposition_budget_top5_media1.json
```

#### MiMo-V2.5 second-reader stage pilot

The primary reader is Gemini 3.5 (in `stage_decomposition.json`); MiMo-V2.5,
the answerability probe, replays the same conditions as the second reader to
quantify the probe-selection bias.

```bash
for COND in gold_symbolic_facts gold_raw_evidence gold_generic_text gold_structured_text retrieved_raw_all_delivered retrieved_raw_budgeted end_to_end; do
  conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition "$COND" --config evaluation/configs/stage_decomposition_reader_mimo.json --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/stage_decomposition/$COND" || break
done
```

### Experiment 3

- [ ] `image_raw`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_raw
```

- [ ] `image_ocr_only`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_ocr_only
```

- [ ] `image_object_only`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_object_only
```

- [ ] `image_spatial_description`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_spatial_description
```

- [ ] `video_raw`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_raw
```

- [ ] `video_sparse_ordered_frames`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_sparse_ordered_frames
```

- [ ] `video_shuffled_frames`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_shuffled_frames
```

- [ ] `video_ordered_event_count_description`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_ordered_event_count_description
```

- [ ] `audio_raw`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_raw
```

- [ ] `audio_asr`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr
```

- [ ] `audio_asr_prosody`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr_prosody
```

#### MiMo-V2.5 second-reader modality pilot

```bash
for COND in image_raw image_ocr_only image_object_only image_spatial_description video_raw video_sparse_ordered_frames video_shuffled_frames video_ordered_event_count_description audio_raw audio_asr audio_asr_prosody; do
  conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition "$COND" --config evaluation/configs/modality_interventions_reader_mimo.json --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/modality_interventions/$COND" || break
done
```

### Experiment 4

- [ ] `flat_rag_caption_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_caption_text
```

- [ ] `flat_rag_omni_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_omni_text
```

- [ ] `flat_rag_native_mm`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_native_mm
```

- [ ] `amem_caption_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_caption_text
```

- [ ] `amem_omni_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_omni_text
```

- [ ] `memoryos_caption_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_caption_text
```

- [ ] `memoryos_omni_text`

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_omni_text
```

### Paper Results

```bash
conda run --no-capture-output -n mm-atlas python evaluation/experiments/build_paper_results.py
```

Use `--qid-manifest evaluation/manifests/controlled_pilot_120.json` on
inference commands for the fixed 120-question pilot. Paper aggregation takes
its pilot paths from `evaluation/configs/paper_results.json`; it does not accept
a QID-manifest flag. Every pilot inference and judge command is listed
separately in the runbook.

---

## Protocol runner (recommended)

For one-command-per-experiment execution and result tracking, use
[EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md). The sections below retain the
backend-specific reference commands.

The protocol runner uses `omni-emb` for RAG/ReAct, `mm-atlas` for protocol
checks, `a-mem` for A-MEM, and `memory_os` for MemoryOS. E2–E4 controlled
experiment inference also uses `omni-emb`. It uses existing persona stores,
the canonical 923-item view, and environment-based API credentials. It never
rebuilds an index or memory store.

```bash
# Inspect every command without API calls
bash evaluation/run_protocol_experiments.sh --suite all --dry-run

# One-persona plumbing gate
bash evaluation/run_protocol_experiments.sh --suite smoke \
  --variants caption_text

# Full Naive RAG + one-search parity gate
bash evaluation/run_protocol_experiments.sh --suite parity

# Full repaired adaptive ReAct runs
bash evaluation/run_protocol_experiments.sh --suite adaptive

# A-MEM and MemoryOS over both supported text representations
bash evaluation/run_protocol_experiments.sh --suite memory \
  --variants caption_text,omni_text

# Checks and every supported backend
bash evaluation/run_protocol_experiments.sh --suite all
```

Set `EVAL_OUTPUT_ROOT`, `EVAL_VARIANTS`, `CONDA_ENV`, `AMEM_CONDA_ENV`, or
`MEMORYOS_CONDA_ENV` to override the defaults. Use `--resume` only for
RAG/ReAct when the target output already contains `details.jsonl`.

---

## A-MEM

### Build index

```bash
conda activate a-mem

# Caption-text index
python evaluation/amem/build_index.py --config evaluation/amem/config_caption.json

# Omni-text index
python evaluation/amem/build_index.py --config evaluation/amem/config_omni.json

# Single persona (dry run)
python evaluation/amem/build_index.py --config evaluation/amem/config_caption.json \
  --max-personas 1 --build-workers 1
```

### Query / QA evaluation

```bash
conda activate a-mem

python evaluation/amem/query_index.py \
  --config evaluation/amem/query_caption_config.json

# Single persona
python evaluation/amem/query_index.py \
  --config evaluation/amem/query_caption_config.json \
  --persona-filter 0083bb89 --query-workers 1
```

## MemoryOS — Evaluate Existing Stores

```bash
conda activate memory_os

python evaluation/memoryos/eval_questions.py \
  --config evaluation/memoryos/memoryos_eval_config_caption.json

python evaluation/memoryos/eval_questions.py \
  --config evaluation/memoryos/memoryos_eval_config_omni.json
```

---

## RAG — Build Indexes

```bash
conda activate omni-emb

# Caption→Text  (VL caption + ASR → text embedding)
python evaluation/rag/build_index_caption_text.py

# Omni→Text    (omni model captions → text embedding)
python evaluation/rag/build_index_omni_text.py

# Omni-MM      (native multimodal → jina embedding)
python evaluation/rag/build_index_omni_mm.py

# Smoke test
python evaluation/rag/build_index_caption_text.py --max-personas 1
```

## RAG — Evaluate

```bash
# Caption→Text index
python evaluation/rag/eval_rag_text.py

# Omni→Text index
python evaluation/rag/eval_rag_text.py \
  --config evaluation/rag/config_eval_omni_text_deepseekv4flash.json

# Omni-MM index
python evaluation/rag/eval_rag_mm.py

# Smoke test
python evaluation/rag/eval_rag_text.py --max-personas 1
```

Configs: `evaluation/rag/config_*.json`

---

## ReAct — Evaluate

```bash
# Caption→Text index
python evaluation/react/eval_rag_react.py \
  --config evaluation/react/config_eval_react_caption_text.json

# Omni→Text index
python evaluation/react/eval_rag_react.py \
  --config evaluation/react/config_eval_react_omni_text.json

# Omni-MM index
python evaluation/react/eval_rag_react.py \
  --config evaluation/react/config_eval_react_omni_mm_gemini.json

# Deterministic one-search parity mode
python evaluation/react/eval_rag_react.py \
  --config evaluation/react/config_eval_react_caption_text.json \
  --react-mode one_search_parity

# Smoke test
python evaluation/react/eval_rag_react.py \
  --config evaluation/react/config_eval_react_caption_text.json \
  --react-mode adaptive_react --max-personas 1 --query-workers 1
```

Configs: `evaluation/react/config_eval_react_*.json`

---

## LLM-as-Judge

```bash
# Judge the default eval output
python evaluation/judge/eval_judge.py

# Judge another eval run
python evaluation/judge/eval_judge.py --input /path/to/details.jsonl

# Smoke test (10 items)
python evaluation/judge/eval_judge.py --max-items 10
```

Config: `evaluation/judge/config_judge.json`

---

## Output Format

All eval scripts write two files per run:

### `details.jsonl`
One JSON line per question: `question_id`, `question_type`, `answerability`, `question`, `expected_answer`, `answer_aliases`, `predicted_answer`, `evaluation`, `retrieval_recall`, `retrieved`, `chain_description`, `multihop_chain`, `evidence_plan`, `latency_seconds`.
**ReAct only**: also `react_trace` and `total_iterations`.
**Judge**: adds `judge_verdict`, `judge_reasoning`, `em`, `f1`, `bleu1`.

### `summary.json`
Aggregated metrics: `overall_accuracy`, `accuracy_answerable`, `accuracy_unanswerable`, `per_question_type`, `em`, `f1`, `bleu1` (judge only), elapsed time.
