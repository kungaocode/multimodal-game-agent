"""Donation finite state machine (phase 10).

Flow:
    VILLAGE -> MESSAGE_LIST -> REQUEST_DETAIL -> DONATING -> VILLAGE

Signals (from detector + OCR):
    message_list_button: opens the clan message list
    request_entry:       a "request troops" entry in the message list
    reinforce_button:    green "donate/reinforce" button on a request
    close_button:        closes the message list / dialog
    confirm_button:      confirms donation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from state.game_state import Resources


class DonationState(str, Enum):
    VILLAGE = "村庄待机"
    MESSAGE_LIST = "捐兵-消息列表"
    REQUEST_FOUND = "捐兵-已检测到需求"
    REQUEST_DETAIL = "捐兵-请求详情"
    DONATING = "捐兵-增援中"
    BLOCKED = "捐兵-资源不足阻塞"
    DONE = "捐兵-完成"


@dataclass
class DonationSignals:
    """Perception signals for the donation FSM."""

    message_list_button: tuple[int, int] | None = None
    request_entry: tuple[int, int] | None = None
    reinforce_button: tuple[int, int] | None = None
    close_button: tuple[int, int] | None = None
    confirm_button: tuple[int, int] | None = None
    troop_bar_coords: tuple[int, int] | None = None
    troop_type: str | None = None  # e.g. "气球兵" (default if None)
    more_requests: bool | None = None  # 捐完一单后是否还有下一单（None=未知/按旧行为回村庄）
    request_cost: int = 0
    request_resource_type: str = "elixir"
    required_amount: int | None = None  # 覆盖 request_cost，用于需求方给出含缓冲的目标
    resources: Resources | None = None
    troop_available: bool = True


@dataclass(frozen=True)
class DonationSupply:
    enough: bool
    required: int
    available: int
    missing: int


@dataclass
class DonationAction:
    kind: str  # tap / wait / stop
    target: str | None = None
    coords: tuple[int, int] | None = None

    def __repr__(self) -> str:
        if self.coords:
            return f"DonationAction({self.kind}, {self.target}, {self.coords})"
        return f"DonationAction({self.kind}, {self.target})"


class DonationFSM:
    """Donation state machine: input perception signals, output actions."""

    def __init__(self) -> None:
        self.state = DonationState.VILLAGE
        self.history: list[str] = [self.state.value]
        self.troop_selected = False
        self.blocked_request: dict[str, int | str] | None = None

    def step(self, signals: DonationSignals) -> DonationAction:
        if self.state is DonationState.VILLAGE:
            if signals.message_list_button is not None:
                return self._transit(
                    DonationState.MESSAGE_LIST,
                    DonationAction("tap", "消息列表按钮", signals.message_list_button),
                )
            return DonationAction("wait", "等待打开部落面板/消息列表")

        if self.state is DonationState.MESSAGE_LIST:
            if signals.close_button is not None and signals.request_entry is None:
                return self._transit(
                    DonationState.VILLAGE,
                    DonationAction("tap", "关闭按钮", signals.close_button),
                )
            if signals.request_entry is not None:
                self.state = DonationState.REQUEST_FOUND
                supply = self._supply(signals)
                if supply.enough:
                    return self._transit(
                        DonationState.REQUEST_DETAIL,
                        DonationAction("tap", "请求条目", signals.request_entry),
                    )
                self.blocked_request = {
                    "resource_type": signals.request_resource_type,
                    "cost": signals.request_cost,
                    "required": supply.required,
                    "available": supply.available,
                    "missing": supply.missing,
                }
                return self._transit(
                    DonationState.BLOCKED,
                    DonationAction("wait", "捐兵-资源不足阻塞"),
                )
            return DonationAction("wait", "等待识别请求条目")

        if self.state is DonationState.BLOCKED:
            if signals.request_entry is not None:
                supply = self._supply(signals)
                if supply.enough:
                    self.blocked_request = None
                    return self._transit(
                        DonationState.REQUEST_DETAIL,
                        DonationAction("tap", "请求条目", signals.request_entry),
                    )
                self.blocked_request = {
                    "resource_type": signals.request_resource_type,
                    "cost": signals.request_cost,
                    "required": supply.required,
                        "available": supply.available,
                        "missing": supply.missing,
                    }
            elif signals.message_list_button is not None:
                # BLOCKED 后可能已离开捐兵面板去补资源；回到村庄再看到入口时
                # 自动重新打开消息列表，继续等资源足够后捐兵。
                return self._transit(
                    DonationState.MESSAGE_LIST,
                    DonationAction("tap", "消息列表按钮", signals.message_list_button),
                )
            return DonationAction("wait", "等待资源补充至捐赠需求")

        if self.state is DonationState.REQUEST_DETAIL:
            if signals.reinforce_button is not None:
                # 每个请求进入增援弹窗都要重新选兵种，不能沿用上一单的选择状态。
                self.troop_selected = False
                return self._transit(
                    DonationState.DONATING,
                    DonationAction("tap", "增援按钮", signals.reinforce_button),
                )
            if signals.close_button is not None:
                return self._transit(
                    DonationState.VILLAGE,
                    DonationAction("tap", "关闭按钮", signals.close_button),
                )
            return DonationAction("wait", "等待增援按钮")

        if self.state is DonationState.DONATING:
            if signals.confirm_button is not None and self.troop_selected:
                return self._transit(
                    DonationState.DONE,
                    DonationAction("tap", "捐赠确认按钮", signals.confirm_button),
                )
            if not self.troop_selected and signals.troop_bar_coords is not None:
                self.troop_selected = True
                return DonationAction(
                    "tap", "兵种选择栏", signals.troop_bar_coords
                )
            troop = signals.troop_type or "气球兵"
            return DonationAction("wait", f"等待兵种选择栏，目标兵种 {troop}")

        if self.state is DonationState.DONE:
            if signals.close_button is not None:
                if signals.more_requests is True:
                    return self._transit(
                        DonationState.MESSAGE_LIST,
                        DonationAction("tap", "关闭按钮", signals.close_button),
                    )
                return self._transit(
                    DonationState.VILLAGE,
                    DonationAction("tap", "关闭按钮", signals.close_button),
                )
            return DonationAction("wait", "捐兵完成，等待关闭")

        return DonationAction("stop", "已停止")

    def _transit(self, new_state: DonationState, action: DonationAction) -> DonationAction:
        self.state = new_state
        if self.history[-1] != new_state.value:
            self.history.append(new_state.value)
        return action

    def reset(self) -> None:
        self.state = DonationState.VILLAGE
        self.history = [self.state.value]
        self.troop_selected = False
        self.blocked_request = None

    def _supply(self, signals: DonationSignals) -> DonationSupply:
        required = (
            int(signals.required_amount)
            if signals.required_amount is not None
            else int(signals.request_cost)
        )
        if not signals.troop_available:
            return DonationSupply(False, required, 0, max(0, required))
        if required <= 0:
            return DonationSupply(True, 0, 0, 0)
        if signals.resources is None:
            # 未读到自家资源时不武断阻塞，避免 OCR 漏读造成误停。
            return DonationSupply(True, required, 0, 0)
        available = getattr(signals.resources, signals.request_resource_type, 0)
        missing = max(0, required - available)
        return DonationSupply(available >= required, required, available, missing)
