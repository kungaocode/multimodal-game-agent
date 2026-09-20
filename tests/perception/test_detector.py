"""Tests for YOLO detector wrapper."""

import pytest


def test_detector_requires_ultralytics():
    pytest.importorskip("ultralytics")
    from perception.detector import Detector

    detector = Detector("yolov8n.pt")
    assert detector.model_path == "yolov8n.pt"
