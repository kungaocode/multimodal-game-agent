"""OCR reader based on PaddleOCR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    confidence: float
    bbox: list[tuple[int, int]]  # four corners


class OCRReader:
    """Wrap PaddleOCR for text recognition in screenshots."""

    def __init__(self, lang: str = "ch", use_gpu: bool = False, **kwargs: Any) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self._kwargs = kwargs
        self._engine: Any | None = None

    def _load(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "paddleocr is not installed. Run: pip install paddleocr"
                ) from exc
            self._engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
                **self._kwargs,
            )
        return self._engine

    def read_text(self, image: np.ndarray | Image.Image | Path | str) -> list[OCRResult]:
        """Recognize all text regions in the image."""
        engine = self._load()
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        result = engine.ocr(image, cls=True)
        outputs: list[OCRResult] = []
        if result is None or result[0] is None:
            return outputs
        for line in result[0]:
            if line is None:
                continue
            bbox, (text, conf) = line
            outputs.append(
                OCRResult(
                    text=str(text),
                    confidence=float(conf),
                    bbox=[(int(p[0]), int(p[1])) for p in bbox],
                )
            )
        return outputs

    def extract_numbers(self, image: np.ndarray | Image.Image | Path | str) -> list[tuple[int | float, OCRResult]]:
        """Extract numeric strings from OCR results."""
        results = self.read_text(image)
        numbers: list[tuple[int | float, OCRResult]] = []
        for r in results:
            cleaned = re.sub(r"[^0-9]", "", r.text)
            if cleaned:
                numbers.append((int(cleaned), r))
        return numbers
