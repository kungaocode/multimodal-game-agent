"""Tests for decision/coach.py —— resource_buildings 解析/过滤 + 提示词规则。

聚焦 §4.3：教练输出全部可见资源建筑回填算法；非 6 类/坐标越界/缺失 → 丢弃；
归一化 center 转像素；旧画布像素坐标按 viewed→orig 缩放兜底。
"""

from __future__ import annotations

from decision.coach import CoachReviewer, _parse_resource_buildings, build_coach_prompt


def _reviewer() -> CoachReviewer:
    return CoachReviewer(None)


def _parsed(**kw) -> dict:
    base = {
        "approved": False,
        "corrected_state": "战斗中",
        "corrected_action": {
            "kind": "deploy",
            "target": "金矿",
            "coords": [0.2, 0.3],
            "troop_coords": [0.5, 0.9],
        },
        "missed_objects": [],
        "reasoning": "测试",
    }
    base.update(kw)
    return base


def test_parse_resource_buildings_normalized_to_pixels():
    r = _reviewer()
    parsed = _parsed(
        resource_buildings=[
            {"type": "金矿", "center": [0.25, 0.5]},
            {"type": "圣水瓶", "center": [0.5, 0.25]},
        ]
    )
    verdict = r._parse_verdict(parsed, image_size=(1280, 720))
    assert verdict.resource_buildings == [("金矿", (320, 360)), ("圣水瓶", (640, 180))]


def test_parse_resource_buildings_filters_non_6class():
    """type 不是 6 类资源建筑名 → 丢弃。"""
    r = _reviewer()
    parsed = _parsed(
        resource_buildings=[
            {"type": "金库", "center": [0.5, 0.5]},   # 不在 6 类内
            {"type": "圣水收集器", "center": [0.5, 0.5]},
        ]
    )
    verdict = r._parse_verdict(parsed, image_size=(1280, 720))
    assert verdict.resource_buildings == [("圣水收集器", (640, 360))]


def test_parse_resource_buildings_drops_bad_coords():
    """center 缺失/越界/非数值 → 丢弃；不影响合法项。"""
    r = _reviewer()
    parsed = _parsed(
        resource_buildings=[
            {"type": "金矿", "center": [0.5, -0.2]},    # 归一化坐标越界(负) → 丢弃
            {"type": "金矿"},                            # center 缺失 → 丢弃
            {"type": "储金罐", "center": "abc"},         # 非数值 → 丢弃
            {"type": "储金罐", "center": [0.5, 0.5]},
        ]
    )
    verdict = r._parse_verdict(parsed, image_size=(1280, 720))
    assert verdict.resource_buildings == [("储金罐", (640, 360))]


def test_parse_resource_buildings_legacy_pixel_scaling():
    """教练输出仍是旧画布像素坐标（>1）时按 legacy viewed→orig 缩放。"""
    r = _reviewer()
    parsed = _parsed(
        resource_buildings=[{"type": "金矿", "center": [320, 240]}]
    )
    verdict = r._parse_verdict(
        parsed, image_size=(2560, 1440), viewed_size=(1280, 960)
    )
    assert verdict.resource_buildings == [("金矿", (640, 360))]


def test_parse_resource_buildings_not_a_list_returns_empty():
    r = _reviewer()
    assert _parse_resource_buildings(None, (1280, 720), (1280, 720)) == []
    assert _parse_resource_buildings("oops", (1280, 720), (1280, 720)) == []
    verdict = r._parse_verdict(_parsed(), image_size=(1280, 720))
    assert verdict.resource_buildings == []


def test_parse_verdict_keeps_existing_fields():
    """新增字段不影响既有解析（coords/missed_objects 仍工作）。"""
    r = _reviewer()
    parsed = _parsed(
        resource_buildings=[{"type": "金矿", "center": [0.25, 0.5]}],
        corrected_action={
            "kind": "deploy",
            "target": "圣水收集器",
            "coords": [0.1, 0.9],
            "troop_coords": [0.5, 0.95],
        },
        missed_objects=[{"type": "储金罐", "bbox": [0.1, 0.1, 0.2, 0.2]}],
    )
    verdict = r._parse_verdict(parsed, image_size=(1280, 720))
    assert verdict.corrected_action_target == "圣水收集器"
    assert verdict.corrected_action_coords == (128, 648)
    assert verdict.corrected_troop_coords == (640, 684)
    assert verdict.missed_objects == [{"type": "储金罐", "bbox": [128.0, 72.0, 256.0, 144.0]}]
    assert verdict.resource_buildings == [("金矿", (320, 360))]


def test_build_coach_prompt_has_resource_buildings_schema_and_battle_rules():
    prompt = build_coach_prompt(
        image_size=(1280, 720),
        local_state="战斗中",
        local_action_kind="deploy",
        local_action_target="金矿",
        local_action_coords=(100, 100),
    )
    assert '"resource_buildings"' in prompt
    assert '"center"' in prompt
    # 红「结束战斗」按钮常驻 ≠ 结束信号
    assert "整场常驻" in prompt
    # 提示词不鼓励看到结束按钮就点结束
    assert "绝不因为看到它" in prompt
    # resource_buildings 要求列出全部可见、并给出归一化 center
    assert "全部敌方资源建筑" in prompt
