#!/usr/bin/env python3
"""Rewrite evaluation details.jsonl with base64 media payloads replaced by paths.

Native-MM ReAct rows persist full ``data:*;base64,...`` blobs (and raw audio
bytes) inside ``react_trace`` / ``trace.react`` ``provider_blocks``. That can
inflate a single run to tens of GB and makes ``build_paper_results.py`` load
times impractical.

This script streams a JSONL file and, for every delivery manifest, replaces
each provider block's payload with the corresponding ``delivered`` unit's
``media_path`` (1:1 with ``native_attached`` units). The result is schema-
compatible for judging / paper aggregation (which only bind on answer fields)
while remaining auditable via on-disk media paths.

Examples
--------
# Single file -> sibling .elided.jsonl
python evaluation/scripts/elide_details_media_payloads.py \\
  --input /path/to/details.jsonl

# Whole run dir (details + judged), then swap into place
python evaluation/scripts/elide_details_media_payloads.py \\
  --run-dir /path/to/omni_mm_xiaomi/react_adaptive --install

# Both official native-MM ReAct runs
python evaluation/scripts/elide_details_media_payloads.py --protocol-react --install
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


DATA_URL_PREFIXES = ("data:image/", "data:video/", "data:audio/", "data:application/")

DEFAULT_PROTOCOL_REACT = [
    Path(
        "${MOSAIC_DATA_ROOT}/"
        "evaluation_results/protocol_v1/omni_mm_xiaomi/react_adaptive"
    ),
    Path(
        "${MOSAIC_DATA_ROOT}/"
        "evaluation_results/protocol_v1/omni_mm_gemini_3_5/react_adaptive"
    ),
]


def _expand_data_path(value: str) -> str:
    token = "${MOSAIC_DATA_ROOT}"
    if token not in value:
        return value
    root = os.environ.get("MOSAIC_DATA_ROOT", "").strip().rstrip("/")
    if not root:
        raise SystemExit("Set MOSAIC_DATA_ROOT to the released data directory.")
    return value.replace(token, root)


def _is_data_url(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(DATA_URL_PREFIXES)


def _is_bulky_b64(value: Any, *, min_chars: int = 10_000) -> bool:
    if not isinstance(value, str) or len(value) < min_chars:
        return False
    sample = value[:120].replace("\n", "")
    return all(c.isalnum() or c in "+/=" for c in sample)


def _elide_block(block: Any, media_path: str | None) -> tuple[Any, bool]:
    """Replace base64 payload in one provider block. Returns (block, changed)."""
    if not isinstance(block, dict):
        return block, False
    out = dict(block)
    changed = False
    path = media_path or out.get("media_path")

    image = out.get("image_url")
    if isinstance(image, dict) and _is_data_url(image.get("url")):
        image = dict(image)
        image["url"] = path or "<missing_media_path>"
        out["image_url"] = image
        changed = True

    video = out.get("video_url")
    if isinstance(video, dict) and _is_data_url(video.get("url")):
        video = dict(video)
        video["url"] = path or "<missing_media_path>"
        out["video_url"] = video
        changed = True

    audio = out.get("input_audio")
    if isinstance(audio, dict) and (
        _is_data_url(audio.get("data")) or _is_bulky_b64(audio.get("data"))
    ):
        audio = dict(audio)
        audio["data"] = path or "<missing_media_path>"
        out["input_audio"] = audio
        changed = True

    # Defensive: any other nested data-URL string fields on the block.
    for key, value in list(out.items()):
        if key in {"image_url", "video_url", "input_audio"}:
            continue
        if _is_data_url(value) or _is_bulky_b64(value):
            out[key] = path or "<missing_media_path>"
            changed = True

    if changed:
        if path:
            out["media_path"] = path
        out["payload_elided"] = True
    return out, changed


def elide_delivery(delivery: Any) -> tuple[Any, int]:
    """Elide provider_blocks in a delivery dict using delivered media_paths."""
    if not isinstance(delivery, dict):
        return delivery, 0
    blocks = delivery.get("provider_blocks")
    if not isinstance(blocks, list) or not blocks:
        return delivery, 0

    delivered = delivery.get("delivered") or []
    attached_paths: list[str | None] = []
    if isinstance(delivered, list):
        for unit in delivered:
            if isinstance(unit, dict) and unit.get("native_attached"):
                mp = unit.get("media_path")
                attached_paths.append(str(mp) if mp else None)

    new_blocks: list[Any] = []
    n_elided = 0
    for index, block in enumerate(blocks):
        path = attached_paths[index] if index < len(attached_paths) else None
        new_block, changed = _elide_block(block, path)
        new_blocks.append(new_block)
        if changed:
            n_elided += 1

    if n_elided == 0:
        return delivery, 0
    out = dict(delivery)
    out["provider_blocks"] = new_blocks
    out["payloads_elided"] = True
    return out, n_elided


def elide_obj(obj: Any) -> tuple[Any, int]:
    """Recursively elide delivery provider_blocks anywhere in a row."""
    if isinstance(obj, list):
        total = 0
        out = []
        for item in obj:
            new_item, n = elide_obj(item)
            out.append(new_item)
            total += n
        return out, total
    if not isinstance(obj, dict):
        return obj, 0

    # Delivery manifests: rewrite provider_blocks from sibling delivered paths.
    if isinstance(obj.get("provider_blocks"), list) and "delivered" in obj:
        rewritten, n_blocks = elide_delivery(obj)
        total = n_blocks
        out: dict[str, Any] = {}
        for key, value in rewritten.items():
            if key == "provider_blocks":
                out[key] = value
                continue
            out[key], child_n = elide_obj(value)
            total += child_n
        return out, total

    total = 0
    out = {}
    for key, value in obj.items():
        out[key], n = elide_obj(value)
        total += n
    return out, total


def iter_run_inputs(run_dir: Path) -> list[Path]:
    paths = [run_dir / "details.jsonl", run_dir / "judge" / "details_judged.jsonl"]
    return [path for path in paths if path.is_file()]


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(input_path.name + ".elided")


def elide_file(
    input_path: Path,
    output_path: Path,
    *,
    progress_every: int = 25,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if output_path == input_path:
        raise ValueError("Output path must differ from input path")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")

    n_rows = 0
    n_changed_rows = 0
    n_blocks = 0
    bytes_in = 0
    bytes_out = 0
    t0 = time.time()

    try:
        with input_path.open("rb") as src, tmp_path.open("w", encoding="utf-8") as dst:
            for raw in src:
                if not raw.strip():
                    continue
                bytes_in += len(raw)
                row = json.loads(raw)
                new_row, elided = elide_obj(row)
                n_rows += 1
                if elided:
                    n_changed_rows += 1
                    n_blocks += elided
                    new_row["_media_payloads_elided"] = True
                    new_row["_media_payloads_elided_blocks"] = elided
                line = json.dumps(new_row, ensure_ascii=False, separators=(",", ":"))
                dst.write(line)
                dst.write("\n")
                bytes_out += len(line) + 1
                if progress_every and n_rows % progress_every == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    print(
                        f"  … {n_rows} rows | "
                        f"in={bytes_in/1e9:.2f}GB out={bytes_out/1e6:.1f}MB | "
                        f"{bytes_in/elapsed/1e6:.1f} MB/s in",
                        flush=True,
                    )
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    elapsed = max(time.time() - t0, 1e-6)
    stats = {
        "input": str(input_path),
        "output": str(output_path),
        "rows": n_rows,
        "rows_changed": n_changed_rows,
        "blocks_elided": n_blocks,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "seconds": round(elapsed, 2),
        "compression_ratio": round(bytes_in / max(bytes_out, 1), 2),
    }
    print(
        f"Wrote {output_path} | rows={n_rows} changed={n_changed_rows} "
        f"blocks={n_blocks} | {bytes_in/1e9:.2f}GB -> {bytes_out/1e6:.1f}MB "
        f"({stats['compression_ratio']}x) in {elapsed:.1f}s",
        flush=True,
    )
    return stats


def install_elided(input_path: Path, elided_path: Path, *, keep_original: bool) -> None:
    """Replace input with elided file; optionally keep original beside it."""
    input_path = input_path.resolve()
    elided_path = elided_path.resolve()
    if not elided_path.is_file():
        raise FileNotFoundError(elided_path)
    bak = input_path.with_name(input_path.name + ".with_base64")
    if bak.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing backup: {bak}. "
            "Remove/rename it first."
        )
    os.replace(input_path, bak)
    os.replace(elided_path, input_path)
    print(f"Installed elided file at {input_path}", flush=True)
    print(f"Original moved to {bak}", flush=True)
    if not keep_original:
        bak.unlink()
        print(f"Deleted original backup {bak}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single details*.jsonl to elide")
    src.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Run directory containing details.jsonl (+ optional judge/). Repeatable.",
    )
    src.add_argument(
        "--protocol-react",
        action="store_true",
        help="Elide the two official protocol_v1 native-MM ReAct run dirs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (only with --input). Default: <input>.elided",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="After writing, replace the original with the elided file "
             "(original saved as *.with_base64 unless --discard-original)",
    )
    parser.add_argument(
        "--discard-original",
        action="store_true",
        help="With --install, delete the *.with_base64 backup after swap "
             "(saves disk; irreversible)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N rows (0 disables)",
    )
    return parser.parse_args(argv)


def collect_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    jobs: list[tuple[Path, Path]] = []
    if args.input:
        inp = args.input
        out = args.output or default_output_path(inp)
        jobs.append((inp, out))
        return jobs

    run_dirs: list[Path] = []
    if args.protocol_react:
        run_dirs.extend(Path(_expand_data_path(str(path))) for path in DEFAULT_PROTOCOL_REACT)
    if args.run_dir:
        run_dirs.extend(args.run_dir)

    for run_dir in run_dirs:
        for inp in iter_run_inputs(run_dir):
            jobs.append((inp, default_output_path(inp)))
    if not jobs:
        raise SystemExit("No input files found")
    return jobs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output and not args.input:
        raise SystemExit("--output is only valid with --input")
    if args.discard_original and not args.install:
        raise SystemExit("--discard-original requires --install")

    jobs = collect_jobs(args)
    print(f"Eliding {len(jobs)} file(s)", flush=True)
    all_stats = []
    for inp, out in jobs:
        if not inp.is_file():
            raise SystemExit(f"Missing input: {inp}")
        print(f"\n## {inp}", flush=True)
        stats = elide_file(inp, out, progress_every=args.progress_every)
        all_stats.append(stats)
        if args.install:
            install_elided(
                inp,
                out,
                keep_original=not args.discard_original,
            )

    print("\nDone.", flush=True)
    total_in = sum(s["bytes_in"] for s in all_stats)
    total_out = sum(s["bytes_out"] for s in all_stats)
    print(
        f"Total {total_in/1e9:.2f}GB -> {total_out/1e6:.1f}MB "
        f"across {len(all_stats)} file(s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
