"""Tests for the synthetic YOLO detection data generator."""

from pathlib import Path

import pytest
from PIL import Image

from perception.dataset import DetectionDataset, LabelMap
from perception.synthetic_data import _iou, generate_scenes, load_sprites, make_grass_background


@pytest.fixture
def sprites():
    items_dir = Path("dataset/text/items")
    if not items_dir.exists():
        pytest.skip("crawled dataset not present")
    return load_sprites(items_dir, "dataset/picture")


def test_grass_background_size():
    bg = make_grass_background(width=320, height=240, seed=1)
    assert bg.size == (320, 240)
    assert bg.mode == "RGBA"


def test_iou():
    a = (0, 0, 10, 10)
    b = (5, 5, 15, 15)  # 重叠 5x5 = 25，并集 100+100-25 = 175
    assert abs(_iou(a, b) - 25 / 175) < 1e-6
    assert _iou(a, (20, 20, 30, 30)) == 0.0


def test_generate_scenes_creates_valid_labels(tmp_path, sprites):
    if len(sprites) < 2:
        pytest.skip("need at least 2 sprites")
    label_map = generate_scenes(sprites, tmp_path, num_train=8, num_val=4, seed=7)
    assert len(label_map.names) >= 2

    for split in ("train", "val"):
        ds = DetectionDataset(
            tmp_path / "images" / split, tmp_path / "labels" / split, label_map
        )
        assert len(ds) > 0
        for i in range(len(ds)):
            sample = ds[i]
            assert sample.boxes, f"{split}/{i}: scene has no boxes"
            for _cid, x1, y1, x2, y2 in sample.boxes:
                assert x1 >= 0 and y1 >= 0
                assert x2 <= sample.image_width and y2 <= sample.image_height
                assert x2 > x1 and y2 > y1


def test_generate_scenes_writes_label_map_json(tmp_path, sprites):
    if len(sprites) < 2:
        pytest.skip("need at least 2 sprites")
    generate_scenes(sprites, tmp_path, num_train=4, num_val=2)
    label_map = LabelMap.load(tmp_path / "labels.json")
    assert label_map.names
    # 与训练 CLI 的预期一致：labels.json 可直接传给 train_detector --labels
    assert Path(tmp_path / "labels" / "train").is_dir()
    assert Path(tmp_path / "images" / "train").is_dir()
