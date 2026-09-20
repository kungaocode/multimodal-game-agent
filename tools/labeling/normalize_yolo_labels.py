"""Clamp YOLO labels into the valid normalized coordinate range."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _clamp_line(line: str) -> tuple[str, bool]:
    fields = line.strip().split()
    if len(fields) != 5:
        return line, False
    class_id = int(float(fields[0]))
    cx, cy, w, h = (float(v) for v in fields[1:])
    w = min(max(w, 1e-6), 1.0)
    h = min(max(h, 1e-6), 1.0)
    cx = min(max(cx, w / 2), 1.0 - w / 2)
    cy = min(max(cy, h / 2), 1.0 - h / 2)
    fixed = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return fixed, fixed != line.strip()


def normalize_labels(root: Path) -> int:
    label_root = root / "labels"
    fixed_files = 0
    for label_path in sorted(label_root.rglob("*.txt")):
        changed = False
        output_lines: list[str] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            new_line, was_changed = _clamp_line(line)
            output_lines.append(new_line)
            changed = changed or was_changed
        if changed:
            label_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
            fixed_files += 1
    return fixed_files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize YOLO label coordinates")
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args(argv)
    fixed = normalize_labels(args.dataset)
    logger.info("Normalized %d label files in %s", fixed, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
