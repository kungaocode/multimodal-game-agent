"""Dataset utilities for detection training and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class DetectionSample:
    image_path: Path
    boxes: list[tuple[int, int, int, int, int]]  # class_id, x1, y1, x2, y2 in pixels
    image_width: int
    image_height: int


class LabelMap:
    """Map class IDs to human-readable names and back."""

    def __init__(self, names: dict[int, str]) -> None:
        self.names = dict(names)
        self._id_by_name = {name: cid for cid, name in self.names.items()}

    @classmethod
    def from_list(cls, names: list[str]) -> "LabelMap":
        return cls({i: name for i, name in enumerate(names)})

    def id_for(self, name: str) -> int:
        if name not in self._id_by_name:
            raise KeyError(f"Unknown label: {name}")
        return self._id_by_name[name]

    def name_for(self, class_id: int) -> str:
        return self.names[class_id]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.names, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelMap":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls({int(k): v for k, v in data.items()})

    def to_yolo_names(self) -> dict[int, str]:
        return dict(self.names)


class DetectionDataset:
    """Load images and YOLO-format labels.

    YOLO label files share the image stem and contain one line per box:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
    """

    def __init__(
        self,
        image_dir: Path | str,
        label_dir: Path | str | None = None,
        label_map: LabelMap | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir) if label_dir else self.image_dir
        self.label_map = label_map
        self.image_paths: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            self.image_paths.extend(sorted(self.image_dir.glob(ext)))
        if not self.image_paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> DetectionSample:
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        width, height = image.size
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes: list[tuple[int, int, int, int, int]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, xc, yc, w, h = map(float, parts)
                x1 = int((xc - w / 2) * width)
                y1 = int((yc - h / 2) * height)
                x2 = int((xc + w / 2) * width)
                y2 = int((yc + h / 2) * height)
                boxes.append((int(class_id), x1, y1, x2, y2))
        return DetectionSample(
            image_path=image_path,
            boxes=boxes,
            image_width=width,
            image_height=height,
        )

    def stats(self) -> dict[str, Any]:
        class_counts: dict[int, int] = {}
        for i in range(len(self)):
            sample = self[i]
            for box in sample.boxes:
                class_counts[box[0]] = class_counts.get(box[0], 0) + 1
        return {
            "num_images": len(self),
            "num_boxes": sum(class_counts.values()),
            "class_counts": class_counts,
            "label_map": self.label_map.to_yolo_names() if self.label_map else None,
        }


def build_yolo_dataset_yaml(
    root: Path | str,
    train_dir: Path | str,
    val_dir: Path | str,
    label_map: LabelMap,
    output_path: Path | str = "dataset.yaml",
) -> Path:
    """Generate a YOLOv8 dataset YAML file."""
    root = Path(root).resolve()
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    output = Path(output_path)
    train_rel = train_dir.relative_to(root) if train_dir.is_absolute() else train_dir
    val_rel = val_dir.relative_to(root) if val_dir.is_absolute() else val_dir
    lines = [
        f"path: {root}",
        f"train: {train_rel}",
        f"val: {val_rel}",
        "names:",
    ]
    for cid, name in sorted(label_map.names.items()):
        lines.append(f"  {cid}: {name}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
