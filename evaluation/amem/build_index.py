#!/usr/bin/env python3
"""**Phase 1** — Build A-MEM indexes from Mosaic data and persist to disk.

For each persona this script:
1. Loads the persona JSON file
2. Creates an A-MEM instance backed by **persistent** ChromaDB
3. Digests all conversation messages (embeddings + index)
4. Saves the memory state and metadata to ``{index_dir}/{persona_uuid}/``

Once built, the index can be queried repeatedly with ``query_index.py``
(no re-embedding needed).

Usage::

    conda activate a-mem

    # Build all 70 personas with 5 parallel workers
    python evaluation/amem/build_index.py

    # Smoke-test: build 2 personas
    python evaluation/amem/build_index.py --max-personas 2

    # Target specific personas
    python evaluation/amem/build_index.py --persona-filter 0083bb89
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
# Path wrangling — allow imports from this directory and project root
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _common import (
    load_config,
    load_persona_file,
    build_amem_instance,
    populate_all_messages,
    memories_to_dicts,
)

# We also need PROJECT_ROOT for the config path default
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))


# ---------------------------------------------------------------------------
# Build one persona
# ---------------------------------------------------------------------------


def build_one(filepath: str, amem_config: dict, index_dir: str, pbar=None) -> dict | None:
    """Build and persist the A-MEM index for a single persona.

    Returns a manifest entry dict, or ``None`` on failure.
    """
    try:
        persona, items = load_persona_file(filepath)
    except Exception:
        logger.exception(f"Failed to load {os.path.basename(filepath)}")
        return None

    persona_uuid = persona.get("uuid", os.path.basename(filepath))
    persona_name = persona.get("name", "unknown")
    out_dir = os.path.join(index_dir, persona_uuid)

    # Clean up any stale data from a previous partial build
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()

    # Per-persona log file
    sink_id = logger.add(
        os.path.join(out_dir, "build.log"), level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    )

    try:
        # Build with persistent ChromaDB at out_dir
        system = build_amem_instance(amem_config, persist_dir=out_dir)
        note_registry = populate_all_messages(items, system, pbar=pbar)

        # Serialize in-memory state so query_index.py can reconstruct it
        mem_dicts = memories_to_dicts(system)
        with open(os.path.join(out_dir, "memories.json"), "w") as f:
            json.dump(mem_dicts, f, ensure_ascii=False)

        with open(os.path.join(out_dir, "note_registry.json"), "w") as f:
            json.dump(note_registry, f, ensure_ascii=False)

        with open(os.path.join(out_dir, "items.json"), "w") as f:
            json.dump(items, f, ensure_ascii=False)

        with open(os.path.join(out_dir, "persona.json"), "w") as f:
            json.dump(persona, f, ensure_ascii=False)

        elapsed = time.time() - t0
        logger.info(
            f"[build] {persona_name}: {len(note_registry)} notes "
            f"({len(items)} items) in {elapsed:.1f}s  ->  {out_dir}"
        )

        return {
            "persona_uuid":  persona_uuid,
            "persona_name":  persona_name,
            "file":          os.path.basename(filepath),
            "num_notes":     len(note_registry),
            "num_items":     len(items),
            "path":          out_dir,
            "elapsed":       round(elapsed, 1),
        }
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build A-MEM persistent indexes from Mosaic data"
    )
    p.add_argument(
        "--config", default=os.path.join(HERE, "config.json"),
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
        help="Parallel workers (default from config, or 1)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.max_personas is not None:
        config["max_personas"] = args.max_personas

    index_dir = args.index_dir or config.get("index_dir", config["output_dir"])
    if args.index_dir:
        config["index_dir"] = args.index_dir

    # Logging — ERROR+ to stderr (terminal stays clean for progress bars),
    # INFO+ to a global overview log, DEBUG+ to per-persona {uuid}/build.log.
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
    pattern = os.path.join(data_dir, "stage_2_text_based_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error(f"No files matching stage_2_text_based_*.json in {data_dir}")
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

    amem_config = config.get("amem_config", {})
    build_workers = args.build_workers or config.get("build_workers", 1)

    logger.info("=" * 60)
    logger.info("A-MEM Index Builder")
    logger.info(f"  Data dir:      {data_dir}")
    logger.info(f"  Index dir:     {index_dir}")
    logger.info(f"  Personas:      {len(files)}")
    logger.info(f"  Workers:       {build_workers}")
    logger.info(f"  Evolution:     {amem_config.get('evolution_enabled', False)}")
    logger.info("=" * 60)

    os.makedirs(index_dir, exist_ok=True)
    manifest: list[dict] = []
    failed: list[dict] = []
    t_start = time.time()

    # Pre-load the embedding model BEFORE creating progress bars.
    # SentenceTransformer prints its own "Loading weights" tqdm bar
    # which would otherwise steal position 0 from our Messages bar.
    from _common import _install_st_singleton
    _install_st_singleton(
        config["amem_config"]["model_name"],
        config["amem_config"].get("embedding_device", "cpu"),
    )

    # Pre-count total messages for a global progress bar
    logger.info("Counting messages ...")
    total_msgs_all = 0
    for fp in files:
        _, items = load_persona_file(fp)
        seen = set()
        for item in items:
            for s in item.get("haystack_sessions", []):
                sid = s.get("session_id", "")
                if sid and sid not in seen:
                    seen.add(sid)
                    total_msgs_all += len(s.get("messages", []))
    logger.info(f"Total: {total_msgs_all} messages across {len(files)} personas")

    shared_pbar = tqdm(
        total=total_msgs_all, desc="  Messages", unit="msg",
        position=0, mininterval=0.5, smoothing=0.1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]",
    )

    if build_workers > 1:
        with ThreadPoolExecutor(max_workers=build_workers) as executor:
            future_map = {
                executor.submit(build_one, fp, amem_config, index_dir, shared_pbar): fp
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
                        f"({entry['num_notes']} notes)"
                    )
    else:
        for fp in tqdm(files, desc="  Personas", unit="persona", position=1):
            entry = build_one(fp, amem_config, index_dir, shared_pbar)
            if entry is None:
                failed.append({"file": os.path.basename(fp), "error": "returned None"})
            else:
                manifest.append(entry)
                logger.info(
                    f"  [{entry['persona_name'][:30]}] {entry['num_notes']} notes "
                    f"in {entry['elapsed']}s"
                )

    shared_pbar.close()

    total_elapsed = time.time() - t_start
    total_notes = sum(e["num_notes"] for e in manifest)

    # Write manifest
    manifest_path = os.path.join(index_dir, "manifest.json")
    manifest_data = {
        "index_dir":       index_dir,
        "data_dir":        data_dir,
        "built_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_personas":    len(manifest),
        "num_failed":      len(failed),
        "total_notes":     total_notes,
        "elapsed_seconds": round(total_elapsed, 1),
        "personas":        manifest,
        "failed":          failed,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    logger.success(
        f"Done: {len(manifest)}/{len(files)} personas "
        f"({total_notes} notes) in {total_elapsed:.1f}s"
    )
    logger.info(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
