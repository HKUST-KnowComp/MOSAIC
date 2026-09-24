#!/usr/bin/env python3
"""**Omni-MM** — Build multimodal embedding FAISS index from native stage2 data.

Reads ``stage2/llm_filtered/`` (native multimodal data with media files) and
creates per-persona FAISS indexes using ``nvidia/omni-embed-nemotron-3b``
via ``sentence_transformers`` (NOT vllm, NOT transformers.AutoModel).

Text-only messages are embedded as ``{"text": ...}`` documents.
Multimodal messages are embedded with actual media files attached.

Usage::

    python evaluation/build_index_omni_mm.py
    python evaluation/build_index_omni_mm.py --max-personas 2
    python evaluation/build_index_omni_mm.py --persona-filter 0083bb89
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path wrangling
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from _common import (
    load_config,
    load_persona_file,
    get_omni_embed_model,
    build_multimodal_index_for_persona,
)

PROJECT_ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Omni-MM: Build multimodal embedding FAISS index from native stage2 data"
    )
    p.add_argument(
        "--config", default=os.path.join(HERE, "config_build_omni_mm.json"),
        help="Path to build config JSON",
    )
    p.add_argument(
        "--max-personas", type=int, default=None,
        help="Limit number of persona files",
    )
    p.add_argument(
        "--persona-filter", default=None,
        help="Comma-separated persona UUID substrings",
    )
    p.add_argument(
        "--index-dir", default=None,
        help="Override index storage directory (default from config)",
    )
    p.add_argument(
        "--build-workers", type=int, default=None,
        help="Parallel persona workers (default from config, or 4)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.max_personas is not None:
        config["max_personas"] = args.max_personas

    index_dir = args.index_dir or config.get("index_dir", config.get("output_dir", ""))
    if args.index_dir:
        config["index_dir"] = args.index_dir

    omni_config = config.get("embedding", {})
    batch_size = omni_config.get("batch_size", 8)
    index_type = config.get("index", {}).get("type", "flat")
    model_name = omni_config.get("model_name", "jinaai/jina-embeddings-v5-omni-small")

    build_workers = args.build_workers or config.get("build_workers", 4)

    # Logging: ERROR+ to stderr, INFO+ to global log, DEBUG+ to per-persona
    logger.remove()
    logger.add(
        sys.stderr, level="ERROR", colorize=True,
        format="<level>{level: <8}</level> | {message}",
    )
    os.makedirs(index_dir, exist_ok=True)
    logger.add(
        os.path.join(index_dir, "build.log"), level="INFO",
        format="{time:HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    # Gather persona files
    data_dir = config["data_dir"]
    file_glob = config.get("persona_file_glob", "stage2_with_media_*.json")
    pattern = os.path.join(data_dir, file_glob)
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No files matching {file_glob} in {data_dir}")
        sys.exit(1)

    if args.persona_filter:
        filters = [f.strip() for f in args.persona_filter.split(",")]
        files = [pf for pf in files
                 if any(filt in os.path.basename(pf) for filt in filters)]
        logger.info(f"Persona filter: {len(files)} files remaining")

    max_personas = config.get("max_personas")
    if max_personas and max_personas < len(files):
        import random
        rng = random.Random(config.get("seed", 64))
        files = rng.sample(files, max_personas)
        logger.info(f"Sampled {len(files)} personas (seed={config.get('seed', 64)})")

    # Pre-load omni-embed model BEFORE creating tqdm bars.
    logger.info(f"Loading omni-embed model: {model_name} ...")
    get_omni_embed_model(omni_config)
    logger.info("Model loaded.")

    # Pre-count chunks for progress bar
    logger.info("Counting chunks ...")
    total_chunks = 0
    for fp in files:
        _, items = load_persona_file(fp)
        from _common import deduplicate_sessions, chunk_messages_multimodal
        sessions = deduplicate_sessions(items)
        chunks = chunk_messages_multimodal(sessions)
        total_chunks += len(chunks)
    logger.info(f"Total: {total_chunks} chunks across {len(files)} personas")

    logger.info("=" * 60)
    logger.info("Omni-MM: Multimodal Embedding Index Builder (omni-embed)")
    logger.info(f"  Data dir:      {data_dir}")
    logger.info(f"  Index dir:     {index_dir}")
    logger.info(f"  Personas:      {len(files)}")
    logger.info(f"  Workers:       {build_workers}")
    logger.info(f"  Model:         {model_name}")
    logger.info(f"  Batch size:    {batch_size}")
    logger.info("=" * 60)

    os.makedirs(index_dir, exist_ok=True)
    manifest: list[dict] = []
    failed: list[dict] = []
    t_start = time.time()

    shared_pbar = tqdm(
        total=total_chunks, desc="  Chunks", unit="chunks",
        position=0, mininterval=0.5, smoothing=0.1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    if build_workers > 1:
        with ThreadPoolExecutor(max_workers=build_workers) as executor:
            future_map = {
                executor.submit(
                    build_multimodal_index_for_persona, fp, index_dir,
                    omni_config, batch_size, index_type, shared_pbar,
                ): fp
                for fp in files
            }
            pbar = tqdm(as_completed(future_map), total=len(files),
                        desc="  Personas", unit="persona", position=1)
            for future in pbar:
                fp = future_map[future]
                try:
                    entry = future.result()
                except Exception as e:
                    failed.append({"file": os.path.basename(fp), "error": str(e)})
                    logger.error(f"Build failed: {os.path.basename(fp)}: {e}")
                    continue
                if entry is None:
                    failed.append({"file": os.path.basename(fp), "error": "returned None"})
                else:
                    manifest.append(entry)
                    pbar.set_postfix_str(
                        f"{entry['persona_name'][:20]} "
                        f"({entry['num_chunks']} chunks)"
                    )
    else:
        for fp in tqdm(files, desc="  Personas", unit="persona", position=1):
            entry = build_multimodal_index_for_persona(
                fp, index_dir, omni_config, batch_size, index_type, shared_pbar,
            )
            if entry is None:
                failed.append({"file": os.path.basename(fp), "error": "returned None"})
            else:
                manifest.append(entry)
                logger.info(
                    f"  [{entry['persona_name'][:30]}] "
                    f"{entry['num_chunks']} chunks "
                    f"({entry['num_text']} text, {entry['num_multimodal']} mm) "
                    f"in {entry['elapsed']}s"
                )

    shared_pbar.close()

    total_elapsed = time.time() - t_start
    total_chunks_built = sum(e["num_chunks"] for e in manifest)

    # Write manifest
    manifest_path = os.path.join(index_dir, "manifest.json")
    manifest_data = {
        "index_dir":       index_dir,
        "data_dir":        data_dir,
        "built_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_name":      model_name,
        "num_personas":    len(manifest),
        "num_failed":      len(failed),
        "total_chunks":    total_chunks_built,
        "elapsed_seconds": round(total_elapsed, 1),
        "personas":        manifest,
        "failed":          failed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Done: {len(manifest)}/{len(files)} personas "
        f"({total_chunks_built} chunks) in {total_elapsed:.1f}s"
    )
    logger.info(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
