"""CLI to train a YOLO detector on a YOLO-format dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import LabelMap, build_yolo_dataset_yaml
from .detector import Detector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train YOLO detector on game screenshots")
    parser.add_argument("--root", required=True, help="Dataset root directory")
    parser.add_argument(
        "--train", default="images/train", help="Train images directory relative to root"
    )
    parser.add_argument(
        "--val", default="images/val", help="Validation images directory relative to root"
    )
    parser.add_argument("--labels", required=True, help="JSON file with label map")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="runs/detect/train", help="Output directory")
    args = parser.parse_args(argv)

    label_map = LabelMap.load(Path(args.labels))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = build_yolo_dataset_yaml(
        root=args.root,
        train_dir=args.train,
        val_dir=args.val,
        label_map=label_map,
        output_path=output_dir / "dataset.yaml",
    )
    detector = Detector(args.model)
    # project 在 ultralytics 中会被自动拼成 runs_dir/task/project，必须传绝对路径，
    # 再配合 name/exist_ok 精确落在 --output 目录。
    best = detector.train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=str(output_dir.parent.resolve()),
        name=output_dir.name,
        exist_ok=True,
    )
    print(f"Best model saved to: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
