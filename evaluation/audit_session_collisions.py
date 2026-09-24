#!/usr/bin/env python3
"""Audit persona-level session identity collisions in the canonical view."""

from __future__ import annotations

import argparse
import json

from protocol import audit_persona_session_collisions, load_canonical_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = audit_persona_session_collisions(load_canonical_records(args.dataset_dir))
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    if not args.quiet:
        print(rendered)


if __name__ == "__main__":
    main()
