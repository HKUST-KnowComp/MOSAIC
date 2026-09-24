# Protocol Experiment Runbook

This is the copy-and-paste run list for the repaired evaluation protocol.
Run every command from the repository root:

```bash
cd /path/to/MOSAIC
```

The commands use this result root:

```text
${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

The runner dispatches environments automatically:

| Backend | Conda environment |
|---|---|
| RAG, ReAct, and E2–E4 controlled experiments | `omni-emb` |
| Protocol checks / LLM judge / paper aggregation | `mm-atlas` |
| A-MEM | `a-mem` |
| MemoryOS | `memory_os` |

Before API-backed commands, export the keys required by the selected models:
`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `YUNWU_API_KEY`, and
`OPENAI_API_KEY` (A-MEM's internal note engine uses gpt-4o-mini).

Do not use the grouped `all` or `smoke` suites when you want one command to
correspond to exactly one experiment. The commands below intentionally run
one variant at a time. Mark a full-run checklist item complete only after its
inference command and matching judge command have both completed successfully;
pilot sections track inference and judging separately.

## Run Status Checklist

- [ ] P0-CHECKS
- [ ] RAG-CAPTION-TEXT
- [ ] RAG-OMNI-TEXT
- [ ] RAG-NATIVE-MM-XIAOMI
- [ ] RAG-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] RAG-NATIVE-MM-GEMINI-3.5
- [ ] REACT-PARITY-CAPTION-TEXT
- [ ] REACT-PARITY-OMNI-TEXT
- [ ] REACT-PARITY-NATIVE-MM-XIAOMI
- [ ] REACT-PARITY-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] REACT-PARITY-NATIVE-MM-GEMINI-3.5
- [ ] REACT-ADAPTIVE-CAPTION-TEXT
- [ ] REACT-ADAPTIVE-OMNI-TEXT
- [ ] REACT-ADAPTIVE-NATIVE-MM-XIAOMI
- [ ] REACT-ADAPTIVE-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] REACT-ADAPTIVE-NATIVE-MM-GEMINI-3.5
- [ ] AMEM-CAPTION-TEXT
- [ ] AMEM-OMNI-TEXT
- [ ] MEMORYOS-CAPTION-TEXT
- [ ] MEMORYOS-OMNI-TEXT
- [ ] E2-PREPARE-RETRIEVAL-MANIFEST
- [ ] E2-GOLD-SYMBOLIC-FACTS
- [ ] E2-GOLD-RAW-EVIDENCE
- [ ] E2-GOLD-GENERIC-TEXT
- [ ] E2-GOLD-STRUCTURED-TEXT
- [ ] E2-RETRIEVED-RAW-ALL-DELIVERED
- [ ] E2-RETRIEVED-RAW-BUDGETED
- [ ] E2-END-TO-END
- [ ] E2-BUDGET-TOP10
- [ ] E2-BUDGET-TOP5-MEDIA1
- [ ] E2-SECOND-READER-PILOT
- [ ] E3-IMAGE-RAW
- [ ] E3-IMAGE-OCR-ONLY
- [ ] E3-IMAGE-OBJECT-ONLY
- [ ] E3-IMAGE-SPATIAL-DESCRIPTION
- [ ] E3-VIDEO-RAW
- [ ] E3-VIDEO-SPARSE-ORDERED-FRAMES
- [ ] E3-VIDEO-SHUFFLED-FRAMES
- [ ] E3-VIDEO-ORDERED-EVENT-COUNT-DESCRIPTION
- [ ] E3-AUDIO-RAW
- [ ] E3-AUDIO-ASR
- [ ] E3-AUDIO-ASR-PROSODY
- [ ] E3-SECOND-READER-PILOT
- [ ] E4-FLAT-RAG-CAPTION-TEXT
- [ ] E4-FLAT-RAG-OMNI-TEXT
- [ ] E4-FLAT-RAG-NATIVE-MM
- [ ] E4-AMEM-CAPTION-TEXT
- [ ] E4-AMEM-OMNI-TEXT
- [ ] E4-MEMORYOS-CAPTION-TEXT
- [ ] E4-MEMORYOS-OMNI-TEXT
- [ ] PAPER-RESULTS

## 0. Protocol Check

### P0-CHECKS

No model API calls. This validates the canonical manifest, protocol metrics,
all three environments, and the session collision invariant.

```bash
bash evaluation/run_protocol_experiments.sh --suite checks --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/session_collision_audit.json`

## 1. Naive RAG Baselines

Run these before the matching parity command. Each command writes one
variant-specific `details.jsonl`, `summary.json`, and `run_manifest.json`.

### RAG-CAPTION-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite rag --variants caption_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/rag/`

### RAG-OMNI-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite rag --variants omni_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/rag/`

### RAG-NATIVE-MM-XIAOMI

```bash
bash evaluation/run_protocol_experiments.sh --suite rag --variants omni_mm_xiaomi --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/rag/`

### RAG-NATIVE-MM-GEMINI (optional)

Gemini 3.1-flash-lite is a lite-tier model; this row is demoted to optional
and runs only after every required row completes.

```bash
bash evaluation/run_protocol_experiments.sh --suite rag --variants omni_mm_gemini --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/rag/`

### RAG-NATIVE-MM-GEMINI-3.5

```bash
bash evaluation/run_protocol_experiments.sh --suite rag --variants omni_mm_gemini_3_5 --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/rag/`

## 2. ReAct One-Search Parity Gates

Each command requires the matching RAG result above because it uses
`--reuse-baseline`. Remove `--reuse-baseline` only when you deliberately want
to rerun that RAG baseline as part of the parity command.

Each parity command writes two comparison artifacts. The binding gate is
`parity_comparison.json` (`--ignore-predictions`): query, per-item ranked
retrieval IDs, and delivery signatures must be identical between the RAG
baseline and the parity run. `parity_comparison_strict.json` additionally
compares predicted-answer strings and is informational: hosted reasoners are
not bit-deterministic at temperature 0, so a small divergence rate (about 2%
of items on identical delivered evidence) is expected and is recorded as a
repeated-run variability datum for the robustness section. Mark a parity
checklist item complete when the binding gate reports `"passed": true`.

### REACT-PARITY-CAPTION-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite parity --variants caption_text --reuse-baseline --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_one_search_parity/`

### REACT-PARITY-OMNI-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite parity --variants omni_text --reuse-baseline --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_one_search_parity/`

### REACT-PARITY-NATIVE-MM-XIAOMI

```bash
bash evaluation/run_protocol_experiments.sh --suite parity --variants omni_mm_xiaomi --reuse-baseline --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_one_search_parity/`

### REACT-PARITY-NATIVE-MM-GEMINI (optional)

```bash
bash evaluation/run_protocol_experiments.sh --suite parity --variants omni_mm_gemini --reuse-baseline --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_one_search_parity/`

### REACT-PARITY-NATIVE-MM-GEMINI-3.5

```bash
bash evaluation/run_protocol_experiments.sh --suite parity --variants omni_mm_gemini_3_5 --reuse-baseline --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_one_search_parity/`

## 3. Adaptive ReAct Runs

### REACT-ADAPTIVE-CAPTION-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite adaptive --variants caption_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_adaptive/`

### REACT-ADAPTIVE-OMNI-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite adaptive --variants omni_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_adaptive/`

### REACT-ADAPTIVE-NATIVE-MM-XIAOMI

```bash
bash evaluation/run_protocol_experiments.sh --suite adaptive --variants omni_mm_xiaomi --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_adaptive/`

### REACT-ADAPTIVE-NATIVE-MM-GEMINI (optional)

```bash
bash evaluation/run_protocol_experiments.sh --suite adaptive --variants omni_mm_gemini --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_adaptive/`

### REACT-ADAPTIVE-NATIVE-MM-GEMINI-3.5

```bash
bash evaluation/run_protocol_experiments.sh --suite adaptive --variants omni_mm_gemini_3_5 --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_adaptive/`

## 4. A-MEM Runs

These commands use the prebuilt persona stores and run in `a-mem`.

### AMEM-CAPTION-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite amem --variants caption_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/amem_native_end_to_end/`

### AMEM-OMNI-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite amem --variants omni_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/amem_native_end_to_end/`

## 5. MemoryOS Runs

These commands use the prebuilt persona memories and run in `memory_os`.

### MEMORYOS-CAPTION-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite memoryos --variants caption_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/memoryos_native_end_to_end/`

### MEMORYOS-OMNI-TEXT

```bash
bash evaluation/run_protocol_experiments.sh --suite memoryos --variants omni_text --output-root ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/memoryos_native_end_to_end/`

## 6. Judge Each Completed Run

The judge runs in `mm-atlas` and requires `OPENROUTER_API_KEY`. Run the
matching command after each inference command. Every judge output contains
`details_judged.jsonl` and `summary_judged.json` in a `judge/` subdirectory.

The binding judge for every run is **Qwen3.7-Plus** without thinking
(`evaluation/judge/config_judge.json`); all commands below use it by default.
Two secondary panel judges — GLM-5.2 and Grok-4.5, also without thinking —
score the Track B paper rows for cross-judge agreement in the
JUDGE-PANEL-E1 section after the per-run commands. Only the Qwen3.7-Plus `judge/`
output feeds `build_paper_results.py`; the panel produces agreement
statistics, never alternative headline scores.

Judge status:

- [ ] JUDGE-RAG-CAPTION-TEXT
- [ ] JUDGE-RAG-OMNI-TEXT
- [ ] JUDGE-RAG-NATIVE-MM-XIAOMI
- [ ] JUDGE-RAG-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] JUDGE-RAG-NATIVE-MM-GEMINI-3.5
- [ ] JUDGE-REACT-PARITY-CAPTION-TEXT
- [ ] JUDGE-REACT-PARITY-OMNI-TEXT
- [ ] JUDGE-REACT-PARITY-NATIVE-MM-XIAOMI
- [ ] JUDGE-REACT-PARITY-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] JUDGE-REACT-PARITY-NATIVE-MM-GEMINI-3.5
- [ ] JUDGE-REACT-ADAPTIVE-CAPTION-TEXT
- [ ] JUDGE-REACT-ADAPTIVE-OMNI-TEXT
- [ ] JUDGE-REACT-ADAPTIVE-NATIVE-MM-XIAOMI
- [ ] JUDGE-REACT-ADAPTIVE-NATIVE-MM-GEMINI (optional; 3.1 lite tier demoted)
- [ ] JUDGE-REACT-ADAPTIVE-NATIVE-MM-GEMINI-3.5
- [ ] JUDGE-AMEM-CAPTION-TEXT
- [ ] JUDGE-AMEM-OMNI-TEXT
- [ ] JUDGE-MEMORYOS-CAPTION-TEXT
- [ ] JUDGE-MEMORYOS-OMNI-TEXT
- [ ] JUDGE-PANEL-E1

### Judge RAG Outputs

#### JUDGE-RAG-CAPTION-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/rag/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/rag/judge
```

#### JUDGE-RAG-OMNI-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/rag/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/rag/judge
```

#### JUDGE-RAG-NATIVE-MM-XIAOMI

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/rag/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/rag/judge
```

#### JUDGE-RAG-NATIVE-MM-GEMINI (optional)

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/rag/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/rag/judge
```

#### JUDGE-RAG-NATIVE-MM-GEMINI-3.5

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/rag/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/rag/judge
```

### Judge ReAct Outputs

#### JUDGE-REACT-PARITY-CAPTION-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_one_search_parity/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_one_search_parity/judge
```

#### JUDGE-REACT-PARITY-OMNI-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_one_search_parity/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_one_search_parity/judge
```

#### JUDGE-REACT-PARITY-NATIVE-MM-XIAOMI

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_one_search_parity/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_one_search_parity/judge
```

#### JUDGE-REACT-PARITY-NATIVE-MM-GEMINI (optional)

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_one_search_parity/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_one_search_parity/judge
```

#### JUDGE-REACT-PARITY-NATIVE-MM-GEMINI-3.5

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_one_search_parity/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_one_search_parity/judge
```

#### JUDGE-REACT-ADAPTIVE-CAPTION-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_adaptive/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/react_adaptive/judge
```

#### JUDGE-REACT-ADAPTIVE-OMNI-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_adaptive/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/react_adaptive/judge
```

#### JUDGE-REACT-ADAPTIVE-NATIVE-MM-XIAOMI

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_adaptive/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_xiaomi/react_adaptive/judge
```

#### JUDGE-REACT-ADAPTIVE-NATIVE-MM-GEMINI (optional)

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_adaptive/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini/react_adaptive/judge
```

#### JUDGE-REACT-ADAPTIVE-NATIVE-MM-GEMINI-3.5

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_adaptive/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_adaptive/judge
```

### Judge A-MEM and MemoryOS Outputs

#### JUDGE-AMEM-CAPTION-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/amem_native_end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/amem_native_end_to_end/judge
```

#### JUDGE-AMEM-OMNI-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/amem_native_end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/amem_native_end_to_end/judge
```

#### JUDGE-MEMORYOS-CAPTION-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/memoryos_native_end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/caption_text/memoryos_native_end_to_end/judge
```

#### JUDGE-MEMORYOS-OMNI-TEXT

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/memoryos_native_end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/omni_text/memoryos_native_end_to_end/judge
```

### JUDGE-PANEL-E1 (multi-judge agreement)

One robustness experiment run as a documented loop after all required
JUDGE-* commands above are complete. For each Track B paper row it scores the
same `details.jsonl` with the two secondary judges (GLM-5.2 and Grok-4.5,
both without thinking) and computes pairwise exact band agreement,
quadratic-weighted Cohen's kappa, Spearman rank correlation, and
per-question-type kappa against the primary Qwen3.7-Plus output. Parity runs are
gates, not paper rows, so they are excluded.

One-command runner (auto-skips Track B rows whose primary `judge/` is not
done yet, and skips rows whose panel outputs already exist):

```bash
bash evaluation/judge/run_judge_panel_e1.sh
```

Useful flags: `--dry-run`, `--force`, `--continue-on-error`,
`--output-root PATH`.

Equivalent manual loop:

```bash
ROOT=${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1
for RUN in caption_text/rag omni_text/rag omni_mm_xiaomi/rag omni_mm_gemini_3_5/rag caption_text/react_adaptive omni_text/react_adaptive omni_mm_xiaomi/react_adaptive omni_mm_gemini_3_5/react_adaptive caption_text/amem_native_end_to_end omni_text/amem_native_end_to_end caption_text/memoryos_native_end_to_end omni_text/memoryos_native_end_to_end; do
  conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --config evaluation/judge/config_judge_glm_5_2.json --input "$ROOT/$RUN/details.jsonl" --output-dir "$ROOT/$RUN/judge_glm_5_2" || break
  conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --config evaluation/judge/config_judge_grok_4_5.json --input "$ROOT/$RUN/details.jsonl" --output-dir "$ROOT/$RUN/judge_grok_4_5" || break
  conda run --no-capture-output -n mm-atlas python evaluation/judge/judge_agreement.py "$ROOT/$RUN/judge" "$ROOT/$RUN/judge_glm_5_2" "$ROOT/$RUN/judge_grok_4_5" --output "$ROOT/$RUN/judge_agreement.json" || break
done
```

Result: `judge_glm_5_2/`, `judge_grok_4_5/`, and `judge_agreement.json` inside
every listed run directory. Mark JUDGE-PANEL-E1 complete when all twelve
`judge_agreement.json` files exist.

## Result Tracking

Every completed inference directory should contain:

```text
details.jsonl
run_manifest.json
```

RAG, ReAct, and A-MEM also write `summary.json`. MemoryOS writes
`memoryos_mosaic_results.json`. A parity run additionally writes
`parity_comparison.json`.

Each completed judge directory should contain:

```text
details_judged.jsonl
summary_judged.json
```

Mark the experiment heading as completed only after its expected files exist.

## 7. Experiment 2: WRITE--RETRIEVE--DELIVER--READ

All Experiment 2 inference commands run in `omni-emb` (needed for Jina/peft embeddings on retrieved conditions and the retrieval-manifest prep) and use Gemini 3.5-flash as the
primary fixed reader (chosen because it played no role in dataset
construction; MiMo v2.5 is the answerability probe and serves only as the
second reader below). They require `YUNWU_API_KEY` for the reader and
`OPENROUTER_API_KEY` for the MiMo structured writer and the judge. Do not
combine retrieval-manifest preparation with a subset filter: the shared
manifest must contain all 784 canonical answerable questions before any
retrieved condition runs. The
complete `evaluation/manifests/fixed_rewrite_queries.json` must already exist.
The retrieval artifact stores original-query and production fixed-rewrite
top-50 rankings over the same native index.

### E2-PREPARE-RETRIEVAL-MANIFEST

Prerequisite: `evaluation/manifests/fixed_rewrite_queries.json` must contain a
fixed rewrite for every canonical question before this command runs. That file
is filled lazily by the first complete Naive RAG run (any variant with
`query_policy: fixed_rewrite`), so run RAG-CAPTION-TEXT (or another full RAG
variant) first; the prepare step only validates completeness and fails on
missing rewrites.

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --prepare-retrieval-manifest
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieval_manifest_top50.json`

### E2-GOLD-SYMBOLIC-FACTS

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_symbolic_facts
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_symbolic_facts/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_symbolic_facts/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_symbolic_facts/judge
```

### E2-GOLD-RAW-EVIDENCE

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_raw_evidence
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_raw_evidence/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_raw_evidence/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_raw_evidence/judge
```

### E2-GOLD-GENERIC-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_generic_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_generic_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_generic_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_generic_text/judge
```

### E2-GOLD-STRUCTURED-TEXT

This condition populates the content-addressed structured-writer cache before
the fixed reader runs. Failed cache entries are retried on `--resume`.

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_structured_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_structured_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_structured_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/gold_structured_text/judge
```

### E2-RETRIEVED-RAW-ALL-DELIVERED

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_all_delivered
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_all_delivered/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_all_delivered/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_all_delivered/judge
```

### E2-RETRIEVED-RAW-BUDGETED

Uses the original-query ranking truncated to the configured top 20 units, then
applies the 3-media-per-modality production caps.

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_budgeted/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_budgeted/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/retrieved_raw_budgeted/judge
```

### E2-END-TO-END

Uses the frozen production fixed-rewrite ranking, truncated to top 20 and
packed under the same production delivery caps and reader.

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition end_to_end
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/end_to_end/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/stage_decomposition/end_to_end/judge
```

### Delivery Budget Sweep

Both commands replay the identical frozen original-query ranking and the same
primary Gemini 3.5 reader as `retrieved_raw_budgeted`; only the unit/media
delivery budget differs. Together with E2-RETRIEVED-RAW-ALL-DELIVERED and
E2-RETRIEVED-RAW-BUDGETED they form the paired delivery-loss budget curve
(top-50 all feasible, top-20/3-media, top-10/3-media, top-5/1-media). Requires
E2-PREPARE-RETRIEVAL-MANIFEST.

#### E2-BUDGET-TOP10

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted --config evaluation/configs/stage_decomposition_budget_top10.json
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top10/retrieved_raw_budgeted/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top10/retrieved_raw_budgeted/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top10/retrieved_raw_budgeted/judge
```

#### E2-BUDGET-TOP5-MEDIA1

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted --config evaluation/configs/stage_decomposition_budget_top5_media1.json
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top5_media1/retrieved_raw_budgeted/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top5_media1/retrieved_raw_budgeted/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/budget_sweep/top5_media1/retrieved_raw_budgeted/judge
```

### Second-Reader Replication (Stage Conditions)

#### E2-SECOND-READER-PILOT

This is one robustness experiment, so its seven conditions run as a documented
loop rather than seven checklist entries. It uses MiMo-V2.5 as the second
reader on the fixed 120-QID pilot manifest, reusing the frozen retrieval
manifest and structured-writer cache from the primary Gemini run. Because
MiMo is the answerability probe, these rows quantify the probe-selection
bias: compare MiMo's advantage over Gemini on probe-pass versus human-rescued
items. Requires E2-PREPARE-RETRIEVAL-MANIFEST and `OPENROUTER_API_KEY`.

```bash
for COND in gold_symbolic_facts gold_raw_evidence gold_generic_text gold_structured_text retrieved_raw_all_delivered retrieved_raw_budgeted end_to_end; do
  conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition "$COND" --config evaluation/configs/stage_decomposition_reader_mimo.json --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/stage_decomposition/$COND" || break
done
```

Judge:

```bash
for COND in gold_symbolic_facts gold_raw_evidence gold_generic_text gold_structured_text retrieved_raw_all_delivered retrieved_raw_budgeted end_to_end; do
  conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/stage_decomposition/$COND/details.jsonl" --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/stage_decomposition/$COND/judge" || break
done
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/stage_decomposition/`

### Experiment 2 Pilot

Use the same fixed 120-QID manifest for each gold condition. Retrieved
conditions still require the full shared retrieval manifest prepared above.

Acceptance criterion (guards against the earlier oracle anomaly, where gold
sessions scored below the full haystack): on the pilot,
`gold_raw_evidence` must not score materially below
`retrieved_raw_all_delivered` or `end_to_end`, and `gold_symbolic_facts`
should be the strongest condition. If either expectation is violated, stop
and inspect the packed prompts, delivery manifests, and judge outputs for the
failing items before launching any full 784-item condition.

#### E2-PILOT-GOLD-SYMBOLIC-FACTS

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_symbolic_facts --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_symbolic_facts
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_symbolic_facts/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_symbolic_facts/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_symbolic_facts/`

#### E2-PILOT-GOLD-RAW-EVIDENCE

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_raw_evidence --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_raw_evidence
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_raw_evidence/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_raw_evidence/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_raw_evidence/`

#### E2-PILOT-GOLD-GENERIC-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_generic_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_generic_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_generic_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_generic_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_generic_text/`

#### E2-PILOT-GOLD-STRUCTURED-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition gold_structured_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_structured_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_structured_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_structured_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/gold_structured_text/`

#### E2-PILOT-RETRIEVED-RAW-ALL-DELIVERED

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_all_delivered --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_all_delivered
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_all_delivered/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_all_delivered/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_all_delivered/`

#### E2-PILOT-RETRIEVED-RAW-BUDGETED

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition retrieved_raw_budgeted --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_budgeted
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_budgeted/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_budgeted/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/retrieved_raw_budgeted/`

#### E2-PILOT-END-TO-END

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_stage_decomposition.py --condition end_to_end --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/end_to_end
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/end_to_end/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/end_to_end/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/stage_decomposition/end_to_end/`

## 8. Experiment 3: Modality Interventions

Every command changes one required carrier per `intervention_id` and leaves
all other chain hops fixed. Each result directory contains `leak_audit.json`;
do not report paired effects unless its `passed` field is `true`.

### E3-IMAGE-RAW

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_raw
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_raw/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_raw/judge
```

### E3-IMAGE-OCR-ONLY

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_ocr_only
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_ocr_only/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_ocr_only/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_ocr_only/judge
```

### E3-IMAGE-OBJECT-ONLY

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_object_only
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_object_only/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_object_only/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_object_only/judge
```

### E3-IMAGE-SPATIAL-DESCRIPTION

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_spatial_description
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_spatial_description/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_spatial_description/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/image_spatial_description/judge
```

### E3-VIDEO-RAW

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_raw
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_raw/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_raw/judge
```

### E3-VIDEO-SPARSE-ORDERED-FRAMES

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_sparse_ordered_frames
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_sparse_ordered_frames/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_sparse_ordered_frames/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_sparse_ordered_frames/judge
```

### E3-VIDEO-SHUFFLED-FRAMES

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_shuffled_frames
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_shuffled_frames/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_shuffled_frames/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_shuffled_frames/judge
```

### E3-VIDEO-ORDERED-EVENT-COUNT-DESCRIPTION

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_ordered_event_count_description
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_ordered_event_count_description/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_ordered_event_count_description/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/video_ordered_event_count_description/judge
```

### E3-AUDIO-RAW

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_raw
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_raw/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_raw/judge
```

### E3-AUDIO-ASR

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr/judge
```

### E3-AUDIO-ASR-PROSODY

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr_prosody
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr_prosody/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr_prosody/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/modality_interventions/audio_asr_prosody/judge
```

### Second-Reader Replication (Modality Conditions)

#### E3-SECOND-READER-PILOT

One robustness experiment run as a documented loop over the eleven
intervention conditions with MiMo-V2.5 as the second reader on the fixed
120-QID pilot manifest. The transformation and frame caches are shared with
the primary Gemini run so both readers see byte-identical intervened
evidence. Requires `OPENROUTER_API_KEY`. Report second-reader paired effects
only for conditions whose `leak_audit.json` passed in the canonical run.

```bash
for COND in image_raw image_ocr_only image_object_only image_spatial_description video_raw video_sparse_ordered_frames video_shuffled_frames video_ordered_event_count_description audio_raw audio_asr audio_asr_prosody; do
  conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition "$COND" --config evaluation/configs/modality_interventions_reader_mimo.json --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/modality_interventions/$COND" || break
done
```

Judge:

```bash
for COND in image_raw image_ocr_only image_object_only image_spatial_description video_raw video_sparse_ordered_frames video_shuffled_frames video_ordered_event_count_description audio_raw audio_asr audio_asr_prosody; do
  conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/modality_interventions/$COND/details.jsonl" --output-dir "${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/modality_interventions/$COND/judge" || break
done
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/second_reader/modality_interventions/`

### Experiment 3 Pilot

The question manifest expands to every matching target carrier in those
questions; therefore case counts can differ between image, video, and audio
families.

#### E3-PILOT-IMAGE-RAW

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_raw --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_raw
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_raw/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_raw/`

#### E3-PILOT-IMAGE-OCR-ONLY

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_ocr_only --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_ocr_only
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_ocr_only/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_ocr_only/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_ocr_only/`

#### E3-PILOT-IMAGE-OBJECT-ONLY

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_object_only --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_object_only
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_object_only/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_object_only/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_object_only/`

#### E3-PILOT-IMAGE-SPATIAL-DESCRIPTION

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition image_spatial_description --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_spatial_description
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_spatial_description/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_spatial_description/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/image_spatial_description/`

#### E3-PILOT-VIDEO-RAW

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_raw --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_raw
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_raw/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_raw/`

#### E3-PILOT-VIDEO-SPARSE-ORDERED-FRAMES

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_sparse_ordered_frames --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_sparse_ordered_frames
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_sparse_ordered_frames/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_sparse_ordered_frames/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_sparse_ordered_frames/`

#### E3-PILOT-VIDEO-SHUFFLED-FRAMES

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_shuffled_frames --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_shuffled_frames
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_shuffled_frames/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_shuffled_frames/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_shuffled_frames/`

#### E3-PILOT-VIDEO-ORDERED-EVENT-COUNT-DESCRIPTION

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition video_ordered_event_count_description --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_ordered_event_count_description
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_ordered_event_count_description/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_ordered_event_count_description/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/video_ordered_event_count_description/`

#### E3-PILOT-AUDIO-RAW

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_raw --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_raw
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_raw/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_raw/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_raw/`

#### E3-PILOT-AUDIO-ASR

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr/`

#### E3-PILOT-AUDIO-ASR-PROSODY

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_modality_interventions.py --condition audio_asr_prosody --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr_prosody
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr_prosody/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr_prosody/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/modality_interventions/audio_asr_prosody/`

## 9. Experiment 4: Fixed-Reader Architecture Comparison

First complete the Experiment 1 RAG, A-MEM, and MemoryOS source commands in
Sections 1, 4, and 5. Each source must contain 923 canonical ordered records
and `run_manifest.json`. The fixed-reader script then selects the same 784
answerable QIDs and replays exported context through Gemini 3.5-flash (the
primary fixed reader, which played no role in dataset construction; requires
`YUNWU_API_KEY`). The matched
delivery policy is the first 20 source-ranked units, identical media caps, and
no additional character truncation. Source-system query behavior is retained
because retrieval organization is the architecture under comparison.

### E4-FLAT-RAG-CAPTION-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_caption_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_caption_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_caption_text/judge
```

### E4-FLAT-RAG-OMNI-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_omni_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_omni_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_omni_text/judge
```

### E4-FLAT-RAG-NATIVE-MM

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_native_mm
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_native_mm/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_native_mm/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/flat_rag_native_mm/judge
```

### E4-AMEM-CAPTION-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_caption_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_caption_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_caption_text/judge
```

### E4-AMEM-OMNI-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_omni_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_omni_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/amem_omni_text/judge
```

### E4-MEMORYOS-CAPTION-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_caption_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_caption_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_caption_text/judge
```

### E4-MEMORYOS-OMNI-TEXT

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_omni_text
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_omni_text/`

Judge:

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/fixed_reader/memoryos_omni_text/judge
```

### Experiment 4 Pilot

#### E4-PILOT-FLAT-RAG-CAPTION-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_caption_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_caption_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_caption_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_caption_text/`

#### E4-PILOT-FLAT-RAG-OMNI-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_omni_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_omni_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_omni_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_omni_text/`

#### E4-PILOT-FLAT-RAG-NATIVE-MM

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition flat_rag_native_mm --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_native_mm
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_native_mm/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_native_mm/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/flat_rag_native_mm/`

#### E4-PILOT-AMEM-CAPTION-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_caption_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_caption_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_caption_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_caption_text/`

#### E4-PILOT-AMEM-OMNI-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition amem_omni_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_omni_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_omni_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/amem_omni_text/`

#### E4-PILOT-MEMORYOS-CAPTION-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_caption_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_caption_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_caption_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_caption_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_caption_text/`

#### E4-PILOT-MEMORYOS-OMNI-TEXT

- [ ] Inference
- [ ] Judge

```bash
conda run --no-capture-output -n omni-emb python evaluation/experiments/run_fixed_reader_comparison.py --condition memoryos_omni_text --qid-manifest evaluation/manifests/controlled_pilot_120.json --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_omni_text
```

```bash
conda run --no-capture-output -n mm-atlas python evaluation/judge/eval_judge.py --input ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_omni_text/details.jsonl --output-dir ${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_omni_text/judge
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/pilots/fixed_reader/memoryos_omni_text/`

## 10. Shared Paper Results

Run this only after every required inference directory and its `judge/`
subdirectory are complete. It fails on missing files, incompatible manifests,
wrong QID populations, judge errors, or pairing differences.

### PAPER-RESULTS

```bash
conda run --no-capture-output -n mm-atlas python evaluation/experiments/build_paper_results.py
```

Result: `${MOSAIC_DATA_ROOT}/evaluation_results/protocol_v1/paper_results/`

Primary outputs: `paper_tables.md`, `paper_results.json`, `run_metrics.csv`,
`paired_effects.csv`, `breakdowns.csv`, `stage_failure_funnel.json`,
`retrieval_and_delivery.json`, `delivery_budget_curve.json`,
`reader_replication.json`, `manifest_validation.json`, and `robustness.json`.
