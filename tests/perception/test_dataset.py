"""Tests for detection dataset utilities."""

import tempfile
from pathlib import Path

from PIL import Image

from perception.dataset import DetectionDataset, LabelMap, build_yolo_dataset_yaml


def test_dataset_loads_yolo_labels():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir = Path(tmp) / "images"
        img_dir.mkdir()
        label_dir = Path(tmp) / "labels"
        label_dir.mkdir()
        img = Image.new("RGB", (100, 80), color=(0, 0, 0))
        img_path = img_dir / "shot_001.png"
        img.save(img_path)
        # class 0, centered at (50,40), 20x20 px -> normalized (0.5, 0.5, 0.2, 0.25)
        label_dir.joinpath("shot_001.txt").write_text("0 0.5 0.5 0.2 0.25\n")
        ds = DetectionDataset(image_dir=img_dir, label_dir=label_dir)
        assert len(ds) == 1
        sample = ds[0]
        assert sample.image_path == img_path
        assert sample.boxes == [(0, 40, 30, 60, 50)]
        assert sample.image_width == 100
        assert sample.image_height == 80


def test_label_map_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labels.json"
        label_map = LabelMap.from_list(["gold_mine", "elixir_collector"])
        label_map.save(path)
        loaded = LabelMap.load(path)
        assert loaded.id_for("elixir_collector") == 1
        assert loaded.name_for(0) == "gold_mine"


def test_build_yolo_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "train"
        val = root / "val"
        train.mkdir()
        val.mkdir()
        label_map = LabelMap.from_list(["gold_mine", "elixir_collector"])
        yaml_path = build_yolo_dataset_yaml(
            root=root,
            train_dir=train,
            val_dir=val,
            label_map=label_map,
            output_path=root / "dataset.yaml",  # 必须写到临时目录内，避免污染项目根
        )
        text = yaml_path.read_text()
        assert "path:" in text
        assert "train:" in text
        assert "val:" in text
        assert "gold_mine" in text
        assert "elixir_collector" in text
