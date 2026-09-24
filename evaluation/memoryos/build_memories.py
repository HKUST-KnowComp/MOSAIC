#!/usr/bin/env python3
"""Phase 1: Build MemoryOS memories from MOSAIC persona files and persist to disk.

Usage:
  cd /path/to/MOSAIC
  python evaluation/memoryos/build_memories.py --config evaluation/memoryos/memoryos_eval_config.json
  python evaluation/memoryos/build_memories.py --config ... --persona-filter 0083bb89
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from common import (
    MemoryOSEvalWrapper,
    count_all_pairs,
    ingest_all_sessions,
    load_eval_config,
    load_json,
    save_json,
)


def _preload_embedding_model(config: dict):
    """Preload embedding model once in main thread to avoid race in workers."""
    mcfg = config["memoryos"]
    model_name = mcfg["embedding_model_name"]
    kwargs = mcfg.get("embedding_model_kwargs", {}) or {}
    from utils import get_embedding as _get_emb
    sys.__stderr__.write(f"Preloading embedding model: {model_name} ...\n")
    _get_emb("warmup", model_name=model_name, **kwargs)
    sys.__stderr__.write("Embedding model ready.\n")


def _build_one(filepath: str, api_key: str, config: dict,
               progress_callback=None) -> dict:
    """Build MemoryOS for one persona. Returns build result dict."""
    persona_data = load_json(filepath)
    persona = persona_data.get("persona", {})
    persona_uuid = persona.get("uuid", os.path.basename(filepath))
    persona_name = persona.get("name", "unknown")
    items = persona_data.get("items", [])

    wrapper = MemoryOSEvalWrapper(persona_uuid, api_key, config)

    start = time.time()
    num_pairs = ingest_all_sessions(wrapper, persona_data, progress_callback)
    elapsed = time.time() - start

    return {
        "persona_uuid": persona_uuid,
        "persona_name": persona_name,
        "file": os.path.basename(filepath),
        "num_pairs_ingested": num_pairs,
        "num_items": len(items),
        "ingest_time_seconds": round(elapsed, 1),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MemoryOS memories from MOSAIC data")
    p.add_argument("--config", required=True, help="Path to evaluation config JSON")
    p.add_argument("--workers", type=int, default=None,
                   help="Override ingest_workers from config")
    p.add_argument("--persona-filter", default=None,
                   help="Comma-separated persona UUID substrings (for testing subsets)")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_eval_config(args.config)
    api_key = config["_resolved_api_key"]

    # --- Log redirection ---
    # All MemoryOS verbose output (prints) goes to a log file.
    # tqdm progress bar stays on the terminal via original stderr.
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "build.log")

    # Recycle: truncate on each run
    log_fh = open(log_path, "w", buffering=1)
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh

    try:
        data_dir = config["data_dir"]
        file_glob = config.get("persona_file_glob", "stage_2_text_based_*.json")
        persona_files = sorted(glob.glob(os.path.join(data_dir, file_glob)))

        if not persona_files:
            _orig_stderr.write(f"ERROR: No files matching '{file_glob}' found in {data_dir}\n")
            sys.exit(1)

        # Persona filter
        if args.persona_filter:
            filters = [f.strip() for f in args.persona_filter.split(",")]
            persona_files = [pf for pf in persona_files
                             if any(filt in os.path.basename(pf) for filt in filters)]

        workers = args.workers or config.get("ingest_workers", 5)
        storage_path = config["memoryos"]["data_storage_path"]
        os.makedirs(storage_path, exist_ok=True)

        _orig_stderr.write(f"Building memories for {len(persona_files)} persona(s)\n")
        _orig_stderr.write(f"  Storage: {storage_path}\n")
        _orig_stderr.write(f"  Workers: {workers}\n")
        _orig_stderr.write(f"  Model:   {config['memoryos']['llm_model']}\n")
        _orig_stderr.write(f"  Log:     {log_path}\n\n")

        # Preload embedding model once to avoid race in parallel workers
        _preload_embedding_model(config)

        # --- Pre-scan: count total QA pairs for progress bar ---
        _orig_stderr.write("Counting total message pairs ...\n")
        total_pairs = 0
        file_pair_counts: dict[str, int] = {}
        for pf in persona_files:
            try:
                n = count_all_pairs(pf)
                total_pairs += n
                file_pair_counts[pf] = n
            except Exception:
                file_pair_counts[pf] = 0
        _orig_stderr.write(f"Total: {total_pairs} QA pairs across {len(persona_files)} persona(s)\n\n")

        # --- Build with progress bar ---
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_pairs, file=_orig_stderr, unit="msg",
                        desc="Ingesting", dynamic_ncols=True)
            _use_tqdm = True
        except ImportError:
            _orig_stderr.write("(install tqdm for progress bar: pip install tqdm)\n\n")
            pbar = None
            _use_tqdm = False

        # Thread-safe progress: tqdm.update() is internally locked
        def _on_pair():
            if pbar:
                pbar.update(1)

        built: list[dict] = []
        failed: list[dict] = []

        start = time.time()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for pf in persona_files:
                future = executor.submit(_build_one, pf, api_key, config, _on_pair)
                future_map[future] = pf

            completed = 0
            total = len(persona_files)
            for future in as_completed(future_map):
                pf = future_map[future]
                completed += 1
                try:
                    result = future.result()
                    built.append(result)
                    if pbar:
                        pbar.write(f"[{completed}/{total}] {result['persona_name'][:30]:30s} "
                                   f"{result['num_pairs_ingested']} pairs in {result['ingest_time_seconds']}s")
                except Exception as e:
                    failed.append({"file": pf, "error": traceback.format_exc()})
                    if pbar:
                        pbar.write(f"[{completed}/{total}] FAILED: {os.path.basename(pf)} -- {e}")

        if pbar:
            pbar.close()

        elapsed = time.time() - start
        _orig_stderr.write(f"\nDone: {len(built)} built, {len(failed)} failed ({elapsed:.0f}s)\n")

        # Write manifest
        manifest_path = os.path.join(storage_path, "manifest.json")
        manifest = {
            "built_at": datetime.now().isoformat(),
            "config_path": os.path.abspath(args.config),
            "data_dir": data_dir,
            "storage_path": storage_path,
            "total_built": len(built),
            "total_failed": len(failed),
            "total_pairs": total_pairs,
            "personas": built,
            "failed": failed,
        }
        save_json(manifest_path, manifest)
        _orig_stderr.write(f"Manifest saved to: {manifest_path}\n")

        if failed:
            _orig_stderr.write(f"\n{len(failed)} persona(s) failed.\n")

    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        log_fh.close()


if __name__ == "__main__":
    main()
