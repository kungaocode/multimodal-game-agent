"""Object detector based on YOLOv8."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class Detector:
    """Wrap Ultralytics YOLO for training and inference."""

    def __init__(self, model_path: Path | str = "yolov8n.pt") -> None:
        self.model_path = str(model_path)
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ultralytics is not installed. Run: pip install ultralytics"
                ) from exc
            self._model = YOLO(self.model_path)
        return self._model

    def train(
        self,
        data_yaml: Path | str,
        epochs: int = 100,
        imgsz: int = 640,
        **kwargs: Any,
    ) -> Path:
        """Train the detector on a YOLO-format dataset YAML.

        ultralytics 的 ``Model.train()`` 返回 metrics 字典而非结果对象，
        最佳权重路径需要从 ``trainer.best`` 读取。
        """
        model = self._load()
        model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, **kwargs)
        return Path(model.trainer.best)

    def predict(
        self,
        image: np.ndarray | Image.Image | Path | str,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> list[DetectionResult]:
        """Run inference and return structured detections."""
        model = self._load()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        results = model(image, conf=conf, iou=iou, verbose=False)
        detections: list[DetectionResult] = []
        names = model.names
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                detections.append(
                    DetectionResult(
                        class_id=class_id,
                        class_name=names.get(class_id, str(class_id)),
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections

    def export(self, format: str = "onnx", **kwargs: Any) -> Path:
        """Export the trained model to another format."""
        model = self._load()
        return Path(model.export(format=format, **kwargs))
