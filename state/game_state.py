"""Structured game state models for the agent."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Resources(BaseModel):
    gold: int = 0
    elixir: int = 0
    dark_elixir: int = 0


class ResourceThresholds(BaseModel):
    """资源触发策略：低于 low 才进入打资源环，回升到 resume 才退出。

    阈值不是大模型判断结果，而是固定配置；模型只负责从画面读数。
    """

    gold_low: int = 500_000
    gold_resume: int = 1_500_000
    elixir_low: int = 500_000
    elixir_resume: int = 1_500_000
    dark_low: int = 5_000
    dark_resume: int = 15_000
    donation_safety_buffer: int = 10_000


class ResourceStatus(BaseModel):
    """自己某项资源是否装满（由顶栏数字 vs 容量表得出）。

    detected 表示识别到几个顶栏数字（按金→圣水→黑油顺序）。
    all_full：至少识别到 2 项（金+圣水）且识别到的每项都满才为 True。
    只识别到 1 项时信息不足，不判全满（避免漏读导致误停）。

    amounts 保存同一帧 OCR 读到的原始数值，供资源阈值判定与节能采集验证使用。
    """

    gold_full: bool = False
    elixir_full: bool = False
    dark_full: bool = False
    detected: int = 0
    amounts: Resources = Field(default_factory=Resources)

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
    resource_status: ResourceStatus = Field(default_factory=ResourceStatus)
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
