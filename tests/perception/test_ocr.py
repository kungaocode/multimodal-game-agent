"""Tests for PaddleOCR wrapper."""

import pytest


def test_ocr_requires_paddleocr():
    pytest.importorskip("paddleocr")
    from perception.ocr import OCRReader

    reader = OCRReader(lang="ch", use_gpu=False)
    assert reader.lang == "ch"
