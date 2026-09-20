"""Tests for tools/labeling/ocr_buttons_to_yolo.py — OCR 文本 → 按钮标注。"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from perception.fsm_labels import FSM_ORDER, class_id_for_button_text
from tools.labeling.ocr_buttons_to_yolo import ocr_buttons_to_labels


class FakeOCRResult:
    def __init__(self, text, bbox, conf=0.95):
        self.text = text
        self.bbox = bbox
        self.confidence = conf


class FakeOCR:
    def __init__(self, results):
        self.results = results

    def read_text(self, image):
        return self.results


def test_class_id_for_button_text():
    assert class_id_for_button_text("进攻！") == FSM_ORDER.index("进攻按钮")
    assert class_id_for_button_text("回营") == FSM_ORDER.index("返回按钮")
    assert class_id_for_button_text("搜索对手") == FSM_ORDER.index("搜索对手按钮")
    assert class_id_for_button_text("下一个") == FSM_ORDER.index("下一个按钮")
    assert class_id_for_button_text("结束战斗") == FSM_ORDER.index("结束战斗按钮")
    assert class_id_for_button_text("增援") == FSM_ORDER.index("增援按钮")
    assert class_id_for_button_text("部落消息") is None


def test_ocr_buttons_to_labels_basic(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (1000, 800), (255, 255, 255)).save(img)
    ocr = FakeOCR(
        [
            FakeOCRResult("进攻！", [(100, 100), (200, 100), (200, 140), (100, 140)]),
            FakeOCRResult("部落消息", [(500, 500), (600, 500), (600, 520), (500, 520)]),
        ]
    )
    lines = ocr_buttons_to_labels(img, ocr, pad=10, min_conf=0.5)
    assert lines is not None
    assert len(lines) == 1
    cid, xc, yc, w, h = lines[0]
    assert FSM_ORDER[int(cid)] == "进攻按钮"
    # 中心点应在文本框中心附近
    assert abs(xc - 150 / 1000) < 0.02
    assert abs(yc - 120 / 800) < 0.02


def test_ocr_buttons_no_match_none(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(img)
    ocr = FakeOCR([FakeOCRResult("部落消息", [(10, 10), (20, 10), (20, 20), (10, 20)])])
    assert ocr_buttons_to_labels(img, ocr, pad=5, min_conf=0.5) is None


def test_ocr_low_confidence_filtered(tmp_path):
    img = tmp_path / "a.png"
    Image.new("RGB", (100, 100), (0, 0, 0)).save(img)
    ocr = FakeOCR([FakeOCRResult("增援", [(10, 10), (20, 10), (20, 20), (10, 20)], conf=0.1)])
    assert ocr_buttons_to_labels(img, ocr, pad=5, min_conf=0.5) is None
