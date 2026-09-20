"""从顶栏 OCR 数字判断自己资源是否装满。

部落冲突村庄顶部从左到右依次是：金 / 圣水 / 黑油 / 宝石 四个数字。
OCR 顶栏后按 (行, 列) 排序取前 3 个数字 → [金, 圣水, 黑油]，
与容量表（自己大本营对应的最大容量）对比得出 full 标记。

容量表需要按玩家实际大本营等级/存储数量调整，默认值是常见配置。
    capacities = Capacities(gold=5_500_000, elixir=5_500_000, dark_elixir=40_000)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from state.game_state import ResourceStatus

# 顶栏高度占全屏的比例：高于这条线的数字才认为属于顶栏
TOP_FRACTION = 0.16


@dataclass(frozen=True)
class Capacities:
    """自己资源的存储上限；按实际大本营配置修改。"""

    gold: int = 5_500_000
    elixir: int = 5_500_000
    dark_elixir: int = 40_000


def _iter_texts(entries: Iterable[Any]) -> Iterable[tuple[str, list[tuple[int, int]] | None]]:
    """兼容 OCRResult 对象和 dict（screen_text 项）两种输入。"""
    for e in entries:
        if isinstance(e, dict):
            text = e.get("text", "")
            bbox = e.get("bbox")
        else:
            text = getattr(e, "text", "")
            bbox = getattr(e, "bbox", None)
        if not text:
            continue
        yield str(text), bbox


def topbar_numbers(
    entries: Iterable[Any],
    image_height: int,
    top_fraction: float = TOP_FRACTION,
) -> list[int]:
    """取顶栏区域的纯数字，按 (行, 列) 排序返回。

    顶栏只有 4 个数字（金/圣水/黑油/宝石），其余文字（名字、等级、按钮）
    要么非数字、要么在顶栏之外，都会被过滤。
    """
    top_limit = image_height * top_fraction
    found: list[tuple[int, int, int]] = []  # (cx, cy, value)
    for text, bbox in _iter_texts(entries):
        if bbox:
            ys = [p[1] for p in bbox]
            if max(ys) > top_limit:
                continue  # 整块在顶栏之下
            cx = sum(p[0] for p in bbox) / len(bbox)
            cy = sum(p[1] for p in bbox) / len(bbox)
        else:
            cx = cy = 0
        digits = re.sub(r"[^0-9]", "", text)
        if not digits:
            continue
        found.append((cx, cy, int(digits)))
    # 顶栏是一行，同一行内按列从左到右排序；y 的小抖动不参与排序
    found.sort()
    return [value for _, _, value in found]


def check_resources_full(
    entries: Iterable[Any],
    image_height: int,
    capacities: Capacities | None = None,
) -> ResourceStatus:
    """把顶栏数字与容量表对比，得到每项是否装满。"""
    caps = capacities or Capacities()
    numbers = topbar_numbers(entries, image_height)

    status = ResourceStatus(detected=min(3, len(numbers)))
    if len(numbers) >= 1:
        status.gold_full = numbers[0] >= caps.gold
    if len(numbers) >= 2:
        status.elixir_full = numbers[1] >= caps.elixir
    if len(numbers) >= 3:
        status.dark_full = numbers[2] >= caps.dark_elixir
    return status
