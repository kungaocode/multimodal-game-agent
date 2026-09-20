"""Tests for tools/labeling/llm_samples_to_yolo.py — 回流样本 → YOLO 标注。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tools.labeling.llm_samples_to_yolo import llm_meta_to_yolo, convert, FARM_ORDER


def _make_sample(root: Path, ts: str, objects: list[dict], size=(1280, 720)) -> None:
    img = Image.new("RGB", size, (10, 10, 10))
    img.save(root / f"{ts}.jpg")
    meta = {
        "timestamp": ts,
        "image": f"{ts}.jpg",
        "route": "llm",
        "reasons": [],
        "prompt": "prompt",
        "response": {"description": "d", "objects": objects},
        "usage": {},
    }
    (root / f"{ts}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def _obj(t, x, y, conf=0.9):
    return {"type": t, "coords": [x, y], "confidence": conf}


def test_llm_meta_to_yolo_converts_valid(tmp_path):
    meta = {
        "response": {
            "objects": [
                _obj("金矿", 100, 100, 0.95),
                _obj("进攻按钮", 640, 640, 0.9),
            ]
        }
    }
    img = tmp_path / "a.jpg"
    Image.new("RGB", (1280, 720), (0, 0, 0)).save(img)
    lines = llm_meta_to_yolo(meta, img, min_conf=0.5, box_scale=1.0)
    assert lines is not None
    ids = [int(l[0]) for l in lines]
    assert FARM_ORDER[ids[0]] in ("金矿", "进攻按钮")
    # 归一化坐标中心应与输入一致（1280x720 画布 = 图片尺寸）
    for line in lines:
        cid, xc, yc, w, h = line
        obj = meta["response"]["objects"][ids.index(int(cid))]
        assert abs(xc - obj["coords"][0] / 1280) < 1e-3
        assert abs(yc - obj["coords"][1] / 720) < 1e-3


def test_llm_meta_filters_bad_objects(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (1280, 720), (0, 0, 0)).save(img)
    meta = {
        "response": {
            "objects": [
                _obj("金矿", 100, 100, 0.3),        # 置信度不足
                _obj("圣水收集器", 0, 0, 0.9),      # coords 非法
                _obj("部落城堡", 100, 100, 0.9),    # 不在 farm 8 类
                _obj("暗黑重油钻井", 640, 360, 0.9),  # 合法
            ]
        }
    }
    lines = llm_meta_to_yolo(meta, img, min_conf=0.5, box_scale=1.0)
    assert lines is not None
    assert len(lines) == 1
    assert int(lines[0][0]) == FARM_ORDER.index("暗黑重油钻井")


def test_llm_meta_none_when_no_valid(tmp_path):
    img = tmp_path / "a.jpg"
    Image.new("RGB", (1280, 720), (0, 0, 0)).save(img)
    meta = {"response": {"objects": [_obj("金矿", 0, 0, 0.9)]}}
    assert llm_meta_to_yolo(meta, img, min_conf=0.5, box_scale=1.0) is None
    meta2 = {"response": None}
    assert llm_meta_to_yolo(meta2, img, min_conf=0.5, box_scale=1.0) is None


def test_convert_dedup_and_output(tmp_path):
    src = tmp_path / "samples"
    src.mkdir()
    # 两张内容相同的图（字节级相同 → 去重保留 1）
    _make_sample(src, "a1", [_obj("金矿", 200, 200, 0.9)])
    _make_sample(src, "a2", [_obj("金矿", 200, 200, 0.9)])
    out = tmp_path / "out"
    stats = convert(src, out, min_conf=0.5, box_scale=1.0, dedup=True)
    assert stats["meta_files"] == 2
    assert stats["unique_images"] == 1
    assert stats["converted_images"] == 1
    assert stats["boxes"] == 1
    assert (out / "labels.json").exists()
    assert (out / "dataset.yaml").exists()
    txts = list((out / "labels").glob("*.txt"))
    assert len(txts) == 1
    assert int(txts[0].read_text().split()[0]) == FARM_ORDER.index("金矿")


def test_convert_no_dedup_keeps_all(tmp_path):
    src = tmp_path / "samples"
    src.mkdir()
    _make_sample(src, "b1", [_obj("金矿", 200, 200, 0.9)])
    _make_sample(src, "b2", [_obj("金矿", 200, 200, 0.9)])
    out = tmp_path / "out"
    stats = convert(src, out, min_conf=0.5, box_scale=1.0, dedup=False)
    assert stats["converted_images"] == 2
