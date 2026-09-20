"""Combine detector, OCR and vision model to produce a GameState."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from state.game_state import BuilderStatus, Building, GameState, Resources

from .classifier import BuildingClassifier
from .detector import Detector
from .ocr import OCRReader
from .vision_model import VisionModel


class GameStateExtractor:
    """Pipeline: screenshot -> detector / classifier / OCR / vision model -> GameState."""

    def __init__(
        self,
        detector: Detector | None = None,
        classifier: BuildingClassifier | None = None,
        ocr: OCRReader | None = None,
        vision_model: VisionModel | None = None,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.ocr = ocr
        self.vision_model = vision_model

    def _predict_level(
        self, pil_image: Image.Image, bbox: tuple[int, int, int, int], warnings: list[str]
    ) -> int | None:
        """用分类器对单个检测框裁剪图预测等级；任何失败返回 None（不中断管线）。

        预处理必须与训练一致：Resize 224 -> ToTensor -> ImageNet 归一化。
        """
        if self.classifier is None:
            return None
        try:
            import torch
            from torchvision import transforms

            width, height = pil_image.size
            x1 = max(0, min(bbox[0], width))
            y1 = max(0, min(bbox[1], height))
            x2 = max(0, min(bbox[2], width))
            y2 = max(0, min(bbox[3], height))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = pil_image.crop((x1, y1, x2, y2))
            tf = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            x = tf(crop).unsqueeze(0)
            with torch.no_grad():
                _type_logits, level_pred = self.classifier(x)
            level = round(float(level_pred.item()))
            return max(1, min(self.classifier.max_level, level))
        except Exception as exc:  # noqa: BLE001 - 单框失败只丢该框的等级
            warnings.append(f"classifier: {exc}")
            return None

    def extract(self, image: Any, image_path: str | None = None) -> GameState:
        """Run the full perception pipeline on a single screenshot."""
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")

        state = GameState(raw_screenshot_path=image_path)
        state.timestamp = datetime.now(timezone.utc).isoformat()

        if self.detector:
            try:
                detections = self.detector.predict(pil_image)
                for det in detections:
                    cx = (det.bbox[0] + det.bbox[2]) // 2
                    cy = (det.bbox[1] + det.bbox[3]) // 2
                    level = self._predict_level(pil_image, det.bbox, state.warnings)
                    state.add_building(
                        building_type=det.class_name,
                        position=(cx, cy),
                        level=level,
                        bbox=det.bbox,
                        confidence=det.confidence,
                    )
            except Exception as exc:  # noqa: BLE001 - 单组件失败不应中断整条流水线
                state.warnings.append(f"detector: {exc}")

        if self.ocr:
            try:
                ocr_results = self.ocr.read_text(pil_image)
                state.screen_text = [
                    {"text": r.text, "confidence": r.confidence, "bbox": r.bbox}
                    for r in ocr_results
                ]
                state.resources = self._heuristic_resources(ocr_results)
            except Exception as exc:  # noqa: BLE001
                state.warnings.append(f"ocr: {exc}")

        if self.vision_model:
            try:
                vision_resp = self.vision_model.describe(pil_image)
                state.screen_text.append(
                    {
                        "source": "vision_model",
                        "description": vision_resp.description,
                        "objects": vision_resp.objects,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                state.warnings.append(f"vision_model: {exc}")

        return state

    def _heuristic_resources(self, ocr_results: list[Any]) -> Resources:
        """Best-effort resource extraction from recognized text."""
        resources = Resources()
        for r in ocr_results:
            text = r.text.replace(",", "").replace(" ", "")
            m = re.search(r"(\d+)", text)
            if not m:
                continue
            value = int(m.group(1))
            lowered = r.text.lower()
            if "gold" in lowered or "金币" in r.text:
                resources.gold = value
            elif "elixir" in lowered or "圣水" in r.text:
                resources.elixir = value
            elif "dark" in lowered or "黑油" in r.text:
                resources.dark_elixir = value
        return resources
