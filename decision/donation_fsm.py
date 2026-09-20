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


class DonationState(str, Enum):
    VILLAGE = "村庄待机"
    MESSAGE_LIST = "捐兵-消息列表"
    REQUEST_DETAIL = "捐兵-请求详情"
    DONATING = "捐兵-增援中"
    DONE = "捐兵-完成"


@dataclass
class DonationSignals:
    """Perception signals for the donation FSM."""

    message_list_button: tuple[int, int] | None = None
    request_entry: tuple[int, int] | None = None
    reinforce_button: tuple[int, int] | None = None
    close_button: tuple[int, int] | None = None
    confirm_button: tuple[int, int] | None = None
    troop_type: str | None = None  # e.g. "气球兵" (default if None)


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
                return self._transit(
                    DonationState.REQUEST_DETAIL,
                    DonationAction("tap", "请求条目", signals.request_entry),
                )
            return DonationAction("wait", "等待识别请求条目")

        if self.state is DonationState.REQUEST_DETAIL:
            if signals.reinforce_button is not None:
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
            if signals.confirm_button is not None:
                return self._transit(
                    DonationState.DONE,
                    DonationAction("tap", "捐赠确认", signals.confirm_button),
                )
            # Auto-select default troop if none specified
            troop = signals.troop_type or "气球兵"
            return DonationAction("tap", f"选择兵种-{troop}", None)

        if self.state is DonationState.DONE:
            if signals.close_button is not None:
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
