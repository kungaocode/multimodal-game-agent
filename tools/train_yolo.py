"""Train a YOLO detector against an existing dataset YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from perception.detector import Detector


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train YOLO detector")
    parser.add_argument("--dataset", type=Path, required=True, help="dataset.yaml or dataset root")
    parser.add_argument("--base", default="yolov8n.pt", help="Base weights")
    parser.add_argument("--out", type=Path, default=Path("runs/detect/fsm_v2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or 0")
    args = parser.parse_args(argv)

    data_yaml = args.dataset / "dataset.yaml" if args.dataset.is_dir() else args.dataset
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    detector = Detector(args.base)
    best = detector.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.out.parent.resolve()),
        name=args.out.name,
        exist_ok=True,
    )
    print(f"Training complete. Best weights: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
