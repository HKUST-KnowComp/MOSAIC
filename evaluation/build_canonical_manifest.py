#!/usr/bin/env python3
"""Generate and validate the repository-local canonical QID manifest."""

from __future__ import annotations

import argparse
import json

from protocol import DEFAULT_QID_MANIFEST, build_qid_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=None, help="Canonical 923-item directory")
    parser.add_argument("--output", default=DEFAULT_QID_MANIFEST, help="Manifest output path")
    args = parser.parse_args()
    manifest = build_qid_manifest(args.dataset_dir, args.output)
    print(json.dumps({k: manifest[k] for k in (
        "num_items", "num_answerable", "num_refusals", "num_personas", "question_id_sha256"
    )}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

