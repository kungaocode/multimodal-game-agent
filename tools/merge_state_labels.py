"""Merge one or more state_labels.json files into one training file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge state_labels.json files")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    samples: list[dict] = []
    for path in args.input:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}")
        samples.extend(data)
        print(f"Loaded {len(data):4d} samples from {path}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = sum(1 for sample in samples if sample.get("state"))
    print(f"Wrote {len(samples)} samples ({valid} with states) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
