# MOSAIC evaluation

Evaluation code for MOSAIC, a long-horizon omni-modal memory benchmark. This repository runs the paper protocol: retrieval, delivery, reading, judging, and the controlled stage and modality comparisons.

The released artifact is the frozen 923-item dataset (784 answerable, 139 refusal). This repository does not include the benchmark generator.

## Layout

```text
evaluation/
  protocol.py          shared delivery and scoring contract
  rag/                 caption-text, omni-text, and native multimodal RAG
  react/               one-search parity and adaptive ReAct
  amem/                A-MEM
  memoryos/            MemoryOS
  judge/               LLM judge
  experiments/         stage decomposition, modality interventions, paper figures
  manifests/           canonical question ids and fixed-rewrite queries
  configs/             paper-run configs
```

## Setup

Point `MOSAIC_DATA_ROOT` at the released data directory. That directory is the one that contains `human_filtered_evaluation_results/` and `evaluation_results/`.

```bash
export MOSAIC_DATA_ROOT=/path/to/mosaic_data
```

Configs store paths as `${MOSAIC_DATA_ROOT}/...`. The loaders expand that variable when a run starts. API keys stay in the environment (`DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `YUNWU_API_KEY`, `OPENAI_API_KEY`). Do not put keys in the JSON configs.

MemoryOS runs also need the MemoryOS checkout:

```bash
export MEMORYOS_PATH=/path/to/memoryos-pypi
```

Run commands from this repository root. The command list is `evaluation/EXPERIMENT_RUNBOOK.md`. Conda environment names in that runbook (`omni-emb`, `mm-atlas`, `a-mem`, `memory_os`) are the ones used for the paper runs; override them with `CONDA_ENV` when your environment names differ.

## Data

The canonical question-id list is `evaluation/manifests/canonical_qids.json` (923 ids). Media and persona session files are distributed with the dataset, not in this repository.
