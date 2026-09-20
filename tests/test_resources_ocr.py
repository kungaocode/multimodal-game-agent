"""Tests for top-bar OCR resource-fullness check."""

from perception.resources_ocr import Capacities, check_resources_full, topbar_numbers


def _ocr(text: str, x: int, y: int, w: int = 60, h: int = 24) -> dict:
    """构造 screen_text 字典项（bbox 为四角点）。"""
    return {"text": text, "bbox": [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]}


def test_topbar_numbers_sorts_by_position_within_top_strip():
    entries = [
        _ocr("1,234,567", x=180, y=30),  # 金
        _ocr("宝石图标", x=20, y=30),  # 非数字 → 过滤
        _ocr("2,000,000", x=420, y=28),  # 圣水
        _ocr("25,000", x=660, y=32),  # 黑油
        _ocr("300", x=900, y=30),  # 宝石（第 4 个）
        _ocr("升级费用 5000", x=300, y=300),  # 顶栏之外 → 过滤
    ]
    assert topbar_numbers(entries, image_height=720) == [1234567, 2000000, 25000, 300]


def test_topbar_ignores_below_bar_and_non_numeric():
    entries = [
        _ocr("部落名称", x=10, y=20),
        _ocr("42", x=400, y=200),  # 顶栏之下
    ]
    assert topbar_numbers(entries, image_height=720) == []


def test_check_resources_full_not_full():
    entries = [_ocr("1,234,567", x=180, y=30), _ocr("2,000,000", x=420, y=28)]
    status = check_resources_full(entries, image_height=720)
    assert status.detected == 2
    assert not status.gold_full and not status.elixir_full
    assert not status.all_full


def test_check_resources_full_with_custom_capacity():
    entries = [_ocr("1,000,000", x=180, y=30)]
    caps = Capacities(gold=1_000_000, elixir=5_500_000, dark_elixir=40_000)
    status = check_resources_full(entries, image_height=720, capacities=caps)
    assert status.gold_full
    assert not status.elixir_full  # 未识别到
    assert not status.all_full  # 圣水未满


def test_all_full_when_every_detected_resource_at_capacity():
    entries = [
        _ocr("6,000,000", x=180, y=30),
        _ocr("6,000,000", x=420, y=28),
        _ocr("45,000", x=660, y=32),
    ]
    status = check_resources_full(entries, image_height=720)
    assert status.detected == 3
    assert status.gold_full and status.elixir_full and status.dark_full
    assert status.all_full


def test_all_full_false_when_nothing_detected():
    assert not check_resources_full([], image_height=720).all_full
