"""Structured game state models for the agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Resources(BaseModel):
    gold: int = 0
    elixir: int = 0
    dark_elixir: int = 0


class ResourceStatus(BaseModel):
    """自己某项资源是否装满（由顶栏数字 vs 容量表得出）。

    detected 表示识别到几个顶栏数字（按金→圣水→黑油顺序）。
    all_full：至少识别到 2 项（金+圣水）且识别到的每项都满才为 True。
    只识别到 1 项时信息不足，不判全满（避免漏读导致误停）。
    """

    gold_full: bool = False
    elixir_full: bool = False
    dark_full: bool = False
    detected: int = 0

    @property
    def all_full(self) -> bool:
        flags: list[bool] = []
        if self.detected >= 1:
            flags.append(self.gold_full)
        if self.detected >= 2:
            flags.append(self.elixir_full)
        if self.detected >= 3:
            flags.append(self.dark_full)
        return self.detected >= 2 and bool(flags) and all(flags)


class BuilderStatus(BaseModel):
    total: int = 0
    idle: int = 0


class Building(BaseModel):
    type: str
    level: int | None = None
    position: tuple[int, int] = Field(default=(0, 0), description="Center (x, y)")
    bbox: tuple[int, int, int, int] | None = Field(
        default=None, description="Bounding box as (x1, y1, x2, y2)"
    )
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class GameState(BaseModel):
    resources: Resources = Field(default_factory=Resources)
    builders: BuilderStatus = Field(default_factory=BuilderStatus)
    buildings: list[Building] = Field(default_factory=list)
    screen_text: list[dict[str, Any]] = Field(default_factory=list)
    raw_screenshot_path: str | None = None
    timestamp: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def add_building(
        self,
        building_type: str,
        position: tuple[int, int],
        level: int | None = None,
        bbox: tuple[int, int, int, int] | None = None,
        confidence: float = 0.0,
    ) -> Building:
        building = Building(
            type=building_type,
            level=level,
            position=position,
            bbox=bbox,
            confidence=confidence,
        )
        self.buildings.append(building)
        return building
