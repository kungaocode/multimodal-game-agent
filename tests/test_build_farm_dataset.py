"""Tests for building the farm-scoped (8-class) synthetic dataset."""

from PIL import Image

from perception.dataset import LabelMap
from tools.labeling.build_farm_dataset import FARM_ORDER, filter_synthetic


def _make_scene(root, split, stem, boxes: list[tuple[int, float, float, float, float]]):
    img_dir = root / "images" / split
    lab_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 200), (88, 148, 64)).save(img_dir / f"{stem}.png")
    lines = [f"{cid} {xc:.4f} {yc:.4f} {w:.4f} {h:.4f}" for cid, xc, yc, w, h in boxes]
    lab_dir.joinpath(f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_filter_remaps_resource_ids_and_drops_others(tmp_path):
    # 源 43 类的一小段：0加农炮 1圣水收集器 2储金罐
    src = tmp_path / "src"
    src.mkdir(parents=True)
    LabelMap.from_list(["加农炮", "圣水收集器", "储金罐"]).save(src / "labels.json")
    _make_scene(src, "train", "a", [(0, 0.3, 0.4, 0.1, 0.1), (1, 0.6, 0.4, 0.1, 0.1)])
    _make_scene(src, "train", "b", [(2, 0.5, 0.5, 0.1, 0.1)])

    out = tmp_path / "out"
    stats = filter_synthetic(src, out)

    # 标签图恒为 FARM_ORDER 8 类
    lm = LabelMap.load(out / "labels.json")
    assert lm.names == {i: name for i, name in enumerate(FARM_ORDER)}

    # 圣水收集器 → farm id 0，加农炮被丢弃
    lab_a = (out / "labels" / "train" / "a.txt").read_text(encoding="utf-8")
    lines_a = lab_a.strip().splitlines()
    assert len(lines_a) == 1
    assert lines_a[0].split()[0] == "0"

    # 储金罐 → farm id 3（FARM_ORDER 里它的下标）
    lab_b = (out / "labels" / "train" / "b.txt").read_text(encoding="utf-8")
    assert lab_b.split()[0] == "3"

    assert stats["train"]["per_class"] == {"圣水收集器": 1, "储金罐": 1}
    assert stats["train"]["images"] == 2


def test_filter_keeps_pure_background_scenes_within_limit(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)
    LabelMap.from_list(["加农炮", "圣水收集器"]).save(src / "labels.json")
    _make_scene(src, "val", "pos", [(1, 0.5, 0.5, 0.1, 0.1)])  # 有资源
    _make_scene(src, "val", "bg", [(0, 0.5, 0.5, 0.1, 0.1)])  # 纯背景

    out = tmp_path / "out"
    stats = filter_synthetic(src, out)

    # max_background=1.0：背景不超过正样本时保留，且标注为空文件
    assert stats["val"]["images"] == 2
    assert stats["val"]["background_only"] == 1
    assert (out / "labels" / "val" / "bg.txt").read_text(encoding="utf-8") == ""


def test_filter_max_background_zero_drops_background(tmp_path):
    src = tmp_path / "src"
    src.mkdir(parents=True)
    LabelMap.from_list(["加农炮", "圣水收集器"]).save(src / "labels.json")
    _make_scene(src, "val", "pos", [(1, 0.5, 0.5, 0.1, 0.1)])
    _make_scene(src, "val", "bg", [(0, 0.5, 0.5, 0.1, 0.1)])

    out = tmp_path / "out"
    stats = filter_synthetic(src, out, max_background=0.0)
    assert stats["val"]["images"] == 1
    assert stats["val"]["background_only"] == 1  # 仍然被计数
    assert not (out / "images" / "val" / "bg.png").exists()
