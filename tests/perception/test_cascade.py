"""Tests for perception/cascade.py — 级联感知路由（本地优先 + LLM 兜底）。"""

from pathlib import Path

import pytest
from PIL import Image

from perception.cascade import (
    CASCADE_PROMPT,
    CascadeConfig,
    CascadeExtractor,
    CascadeReport,
    CascadeStats,
    LlmSampleRecorder,
    MODE_AUTO,
    MODE_LLM,
    MODE_LOCAL,
    _llm_objects_to_buildings,
    needs_llm,
)
from state.game_state import GameState


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def predict(self, image):
        return self.detections


class FakeVision:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = 0
        self.last_prompt = None

    def chat(self, image, prompt, include_usage=False):
        self.calls += 1
        self.last_prompt = prompt
        result = dict(self.parsed)
        if include_usage and "usage" in self.parsed:
            result["_usage"] = self.parsed["usage"]
        return result


def _detection(name, cx, cy, conf):
    from perception.detector import DetectionResult

    return DetectionResult(
        class_id=0, class_name=name, confidence=conf, bbox=(cx - 5, cy - 5, cx + 5, cy + 5)
    )


def _extractor(detections, vision=None, config=None, recorder=None):
    return CascadeExtractor(
        FakeDetector(detections),
        None,
        vision,
        config=config,
        recorder=recorder,
    )


# ---------- 路由决策 ----------

def test_needs_llm_empty_state_triggers():
    need, reasons = needs_llm(GameState(), CascadeConfig())
    assert need is True
    assert any("未检出任何建筑" in r for r in reasons)


def test_needs_llm_high_confidence_ok():
    state = GameState()
    state.add_building("金矿", (100, 200), confidence=0.9)
    state.add_building("圣水收集器", (300, 400), confidence=0.85)
    need, reasons = needs_llm(state, CascadeConfig())
    assert need is False
    assert reasons == []


def test_needs_llm_low_confidence_triggers():
    state = GameState()
    state.add_building("金矿", (100, 200), confidence=0.2)
    need, reasons = needs_llm(state, CascadeConfig(min_avg_confidence=0.5))
    assert need is True
    assert any("平均置信度" in r for r in reasons)


def test_needs_llm_missing_key_buildings_triggers():
    state = GameState()
    state.add_building("加农炮", (100, 200), confidence=0.9)  # 防御建筑不算关键
    need, reasons = needs_llm(state, CascadeConfig())
    assert need is True
    assert any("关键建筑" in r for r in reasons)


def test_needs_llm_donation_scene_config():
    # 捐兵场景：关键建筑换成部落城堡
    state = GameState()
    state.add_building("加农炮", (100, 200), confidence=0.9)
    cfg = CascadeConfig(require_key_buildings=("部落城堡",))
    need, reasons = needs_llm(state, cfg)
    assert need is True
    state2 = GameState()
    state2.add_building("部落城堡", (100, 200), confidence=0.9)
    need2, _ = needs_llm(state2, cfg)
    assert need2 is False


# ---------- LLM 结果合并 ----------

def test_llm_objects_to_buildings_merges_valid():
    state = GameState()
    objects = [
        {"type": "金矿", "coords": [100, 200], "confidence": 0.9},
        {"type": "圣水收集器", "coords": [640, 360], "confidence": 0.8},
        {"type": "未知", "coords": [0, 0], "confidence": 0.9},  # 非法坐标
        {"type": "坏数据", "coords": ["a", "b"]},               # 非法类型
        {"coords": [50, 50]},                                   # 缺 type
    ]
    merged = _llm_objects_to_buildings(state, objects)
    assert merged == 2
    assert len(state.buildings) == 2
    assert state.buildings[0].type == "金矿"
    assert state.buildings[0].position == (100, 200)


# ---------- 样本回流 ----------

def test_recorder_saves_meta_and_image(tmp_path):
    rec = LlmSampleRecorder(tmp_path, max_samples=10)
    img = Image.new("RGB", (640, 360), "white")
    path = rec.record(img, "prompt", {"objects": []}, ["测试原因"], usage={"prompt_tokens": 10})
    meta = Path(path)
    assert meta.exists()
    assert meta.with_suffix(".jpg").exists()
    import json

    data = json.loads(meta.read_text(encoding="utf-8"))
    assert data["reasons"] == ["测试原因"]
    assert data["prompt"] == "prompt"
    assert data["usage"]["prompt_tokens"] == 10
    assert rec.count() == 1


def test_recorder_trims_oldest(tmp_path):
    rec = LlmSampleRecorder(tmp_path, max_samples=2)
    img = Image.new("RGB", (32, 32), "white")
    rec.record(img, "p1", None, [])
    rec.record(img, "p2", None, [])
    rec.record(img, "p3", None, [])
    assert rec.count() == 2
    metas = sorted(tmp_path.glob("*.json"))
    assert "p2" in metas[0].read_text(encoding="utf-8") or "p3" in metas[0].read_text(encoding="utf-8")


# ---------- 级联提取器路由 ----------

def test_cascade_auto_uses_local_when_confident():
    vision = FakeVision({"description": "x", "objects": []})
    ext = _extractor(
        [_detection("金矿", 100, 200, 0.9), _detection("圣水收集器", 300, 400, 0.85)],
        vision=vision,
    )
    state, report = ext.extract(Image.new("RGB", (1280, 720), "white"))
    assert report.route == "local"
    assert vision.calls == 0  # 没调 LLM
    assert state.warnings == []


def test_cascade_auto_calls_llm_and_merges():
    parsed = {
        "description": "村庄",
        "objects": [
            {"type": "金矿", "coords": [100, 200], "confidence": 0.9},
            {"type": "圣水收集器", "coords": [640, 360], "confidence": 0.8},
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }
    vision = FakeVision(parsed)
    ext = _extractor([], vision=vision)  # 本地什么都没检出 → 触发 LLM
    state, report = ext.extract(Image.new("RGB", (1280, 720), "white"))
    assert report.route == "llm"
    assert vision.calls == 1
    assert report.llm_calls == 1
    assert report.prompt_tokens == 100
    assert any("未检出任何建筑" in r for r in report.reasons)
    assert len(state.buildings) == 2
    assert any("LLM 兜底补入" in w for w in state.warnings)
    assert report.sample_saved is not None
    assert Path(report.sample_saved).exists()


def test_cascade_mode_llm_forces_call():
    vision = FakeVision({"description": "x", "objects": []})
    ext = _extractor(
        [_detection("金矿", 100, 200, 0.9)],
        vision=vision,
        config=CascadeConfig(mode=MODE_LLM),
    )
    state, report = ext.extract(Image.new("RGB", (1280, 720), "white"))
    assert report.route == "llm"
    assert vision.calls == 1


def test_cascade_mode_local_never_calls_llm():
    vision = FakeVision({"description": "x", "objects": []})
    ext = _extractor([], vision=vision, config=CascadeConfig(mode=MODE_LOCAL))
    state, report = ext.extract(Image.new("RGB", (1280, 720), "white"))
    assert report.route == "local"
    assert vision.calls == 0
    assert "本地模型未检出" in "".join(state.warnings) or state.warnings == []
