"""Tests for tools/eval_state_transition.py — 状态转换识别测试工具。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from perception.detector import DetectionResult
from tools.eval_state_transition import (
    signal_rules_state,
    yolo_signals,
)
from decision.battle_fsm import BattleSignals


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, image, conf=0.25, iou=0.45):
        return self.detections


def _det(name, cx, cy, conf=0.9):
    return DetectionResult(
        class_id=0,
        class_name=name,
        confidence=conf,
        bbox=(cx - 5, cy - 5, cx + 5, cy + 5),
    )


def _img(tmp_path: Path) -> Image.Image:
    img = Image.new("RGB", (100, 100), (255, 255, 255))
    p = tmp_path / "t.png"
    img.save(p)
    return Image.open(p)


def test_yolo_signals_assembles_buttons_and_resources(tmp_path):
    det = FakeDetector(
        [
            _det("进攻按钮", 10, 20),
            _det("金矿", 40, 50),
            _det("圣水收集器", 60, 70),
        ]
    )
    sig = yolo_signals(det, _img(tmp_path))
    assert sig.attack_button == (10, 20)
    assert sig.search_button is None
    assert sorted(sig.enemy_resources) == [
        ("圣水收集器", (60, 70)),
        ("金矿", (40, 50)),
    ]


def test_signal_rules_state():
    assert signal_rules_state(BattleSignals(attack_button=(1, 1))) == "村庄待机"
    assert signal_rules_state(BattleSignals(search_button=(1, 1))) == "搜索中"
    assert signal_rules_state(BattleSignals(return_button=(1, 1))) == "搜索中"
    assert signal_rules_state(BattleSignals(next_button=(1, 1))) == "战斗中"
    assert signal_rules_state(BattleSignals(end_battle_button=(1, 1))) == "战斗中"
    assert signal_rules_state(BattleSignals()) == "未知/等待"


def test_run_eval_end_to_end(tmp_path, monkeypatch):
    """端到端：ground truth + fake 检测器 → 准确率与信号检出。"""
    from tools import eval_state_transition as ev

    # 两张图：村庄待机（有进攻按钮）+ 战斗中（有结束战斗按钮）
    img1 = tmp_path / "village.png"
    img2 = tmp_path / "battle.png"
    Image.new("RGB", (100, 100), (0, 0, 255)).save(img1)
    Image.new("RGB", (200, 200), (255, 0, 0)).save(img2)

    labels = {
        "samples": {
            "village.png": {
                "expected_state": "村庄待机",
                "signals": {"attack_button": True, "return_button": False},
            },
            "battle.png": {
                "expected_state": "战斗中",
                "signals": {"attack_button": False, "end_battle_button": True},
            },
        }
    }
    (tmp_path / "labels.json").write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")

    class SizeAwareDetector(FakeDetector):
        def predict(self, image, conf=0.25, iou=0.45):
            if image.size == (100, 100):  # village.png
                return [_det("进攻按钮", 20, 20)]
            return [_det("结束战斗按钮", 80, 80)]  # battle.png

    def fake_load_detector(weights):
        return SizeAwareDetector([])

    monkeypatch.setattr(ev, "load_detector", fake_load_detector)
    report = ev.run_eval(
        images_dir=tmp_path,
        labels_path=tmp_path / "labels.json",
        weights="whatever.pt",
        use_ocr=False,
        sequence=None,
    )
    sa = report["state_accuracy"]
    assert sa == {"correct": 2, "total": 2, "rate": 1.0}
    assert report["signal_recall"]["attack_button"]["tp"] == 1
    assert report["signal_recall"]["end_battle_button"]["tp"] == 1
    assert report["signal_recall"]["return_button"]["expected"] == 0


def test_run_eval_sequence_drives_fsm(tmp_path, monkeypatch):
    """序列模式：进攻→搜索→战斗→结束，验证完整转换链。"""
    from tools import eval_state_transition as ev

    img1 = tmp_path / "s1.png"
    img2 = tmp_path / "s2.png"
    img3 = tmp_path / "s3.png"
    img4 = tmp_path / "s4.png"
    for p in (img1, img2, img3, img4):
        Image.new("RGB", (100, 100), (0, 255, 0)).save(p)

    def fake_load_detector(weights):
        return FakeDetector([_det("进攻按钮", 50, 50)])

    monkeypatch.setattr(ev, "load_detector", fake_load_detector)
    report = ev.run_eval(
        images_dir=tmp_path,
        labels_path=tmp_path / "labels.json",  # 不存在也行（sequence 不依赖 labels）
        weights="x.pt",
        use_ocr=False,
        sequence=["s1.png", "s2.png", "s3.png", "s4.png"],
    )
    seq = report["sequence"]
    # s1 见进攻按钮 → 搜索中；s2-s4 无信号 → 保持搜索中
    assert seq["history"] == ["村庄待机", "搜索中"]
