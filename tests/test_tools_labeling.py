"""Tests for the semi-automatic labeling tool backend."""

import json

import pytest
from PIL import Image

import tools.labeling.app as label_app
from perception.dataset import LabelMap


@pytest.fixture
def label_map(tmp_path):
    path = tmp_path / "labels.json"
    LabelMap.from_list(["加农炮", "法师塔"]).save(path)
    return path


@pytest.fixture
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    Image.new("RGB", (320, 200), (88, 148, 64)).save(raw / "shot.png")
    return raw


def test_list_images(raw_dir, monkeypatch):
    monkeypatch.setattr(label_app, "RAW_DIR", raw_dir)
    assert label_app._list_images() == ["shot.png"]


def test_resolve_image_blocks_traversal(raw_dir, monkeypatch):
    monkeypatch.setattr(label_app, "RAW_DIR", raw_dir)
    with pytest.raises(Exception):  # noqa: BLE001 - HTTPException
        label_app._resolve_image("../../etc/passwd")


def test_save_reload_roundtrip(raw_dir, label_map, tmp_path, monkeypatch):
    monkeypatch.setattr(label_app, "RAW_DIR", raw_dir)
    monkeypatch.setattr(label_app, "OUT_DIR", tmp_path / "out")
    monkeypatch.setattr(label_app, "LABELS_PATH", label_map)

    ann = label_app.Annotation(
        boxes=[
            label_app.Box(class_id=0, x1=10, y1=20, x2=110, y2=120),
            label_app.Box(class_id=1, x1=200, y1=30, x2=300, y2=130),
        ]
    )
    result = label_app.save_annotation("shot.png", ann)
    assert result["saved"] == 2

    # 落盘为归一化 YOLO 格式
    text = (tmp_path / "out" / "labels" / "shot.txt").read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert len(lines) == 2
    parts = lines[0].split()
    assert len(parts) == 5
    assert parts[0] == "0"

    # 重新加载回像素坐标
    boxes = label_app._saved_boxes("shot.png")
    assert boxes is not None
    assert len(boxes) == 2
    assert (boxes[0].x1, boxes[0].y1, boxes[0].x2, boxes[0].y2) == (10, 20, 110, 120)
