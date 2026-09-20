"""Merge existing YOLO datasets into a single 17-class dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from perception.dataset import LabelMap, build_yolo_dataset_yaml
from perception.fsm_labels import EXTENDED_FSM_ORDER


def _safe_prefix(source: Path) -> str:
    key = f"{source.parent.name}_{source.name}"
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in key)


def _copy_split(source: Path, split: str, prefix: str, out_root: Path) -> int:
    src_img = source / "images" / split
    src_lab = source / "labels" / split
    if not src_img.exists():
        return 0
    dst_img = out_root / "images" / split
    dst_lab = out_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(src_img.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label_path = src_lab / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        new_stem = f"{prefix}_{image_path.stem}"
        dst_image = dst_img / f"{new_stem}{image_path.suffix.lower()}"
        dst_label = dst_lab / f"{new_stem}.txt"
        if dst_image.exists() and dst_label.exists():
            count += 1
            continue
        if not dst_image.exists():
            try:
                os.link(image_path, dst_image)
            except OSError:
                shutil.copy2(image_path, dst_image)
        if not dst_label.exists():
            shutil.copy2(label_path, dst_label)
        count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets")
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("dataset/fsm_v2"))
    args = parser.parse_args(argv)

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary: dict[str, int] = {}
    for source in args.source:
        prefix = _safe_prefix(source)
        n_train = _copy_split(source, "train", prefix, args.out)
        n_val = _copy_split(source, "val", prefix, args.out)
        summary[f"{prefix}_train"] = n_train
        summary[f"{prefix}_val"] = n_val

    label_map = LabelMap.from_list(EXTENDED_FSM_ORDER)
    label_map.save(args.out / "labels.json")
    yaml_path = build_yolo_dataset_yaml(
        root=args.out,
        train_dir="images/train",
        val_dir="images/val",
        label_map=label_map,
        output_path=args.out / "dataset.yaml",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output dataset: {args.out}")
    print(f"Dataset YAML:   {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
