"""捐兵流程确定性模拟器。

模拟五屏：村庄 → 消息列表 → 请求详情 → 增援选择 → 完成。感知输出与
`decision.donation_fsm.DonationSignals` 兼容，可无截图/无模型地闭环验证
「识别 → FSM 决策 → 校验 → 执行 → 下一帧」整条捐兵链路。

过程要点与真实操作一致：
- 必须先点兵种选择栏（选中兵种）再点确认；
- 每个请求捐完后通过关闭按钮回到消息列表，继续下一个请求；
- 全部请求处理完后回到村庄，标记 done。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from decision.donation_fsm import DonationSignals
from executor.simulator import SimulationResult
from state.game_state import Resources


class DonationScreen(str, Enum):
    VILLAGE = "village"
    MESSAGE_LIST = "message_list"
    REQUEST_DETAIL = "request_detail"
    DONATING = "donating"
    DONE = "done"


@dataclass
class DonationRequest:
    """一条待捐请求：消息列表项坐标与需要的兵种。"""

    coords: tuple[int, int]
    troop_type: str = "气球兵"
    cost: int = 0
    resource_type: str = "elixir"


class DonationSimulator:
    """确定性捐兵模拟器：perceive() → DonationSignals，step() 校验动作并转移。"""

    def __init__(
        self,
        requests: list[DonationRequest] | None = None,
        image_size: tuple[int, int] = (1280, 720),
        own_resources: Resources | None = None,
    ) -> None:
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self._initial_resources = self._copy_resources(own_resources or Resources())
        self.requests = [
            DonationRequest(
                (int(r.coords[0]), int(r.coords[1])),
                troop_type=r.troop_type,
                cost=int(r.cost),
                resource_type=str(r.resource_type),
            )
            for r in (requests or [DonationRequest((640, 300))])
        ]
        w, h = self.image_size
        self.message_list_button = (w // 2, h - 90)
        self.close_button = (w - 70, 70)
        self.reinforce_button = (w // 2, int(h * 0.82))
        self.confirm_button = (w // 2, int(h * 0.55))
        self.troop_bar_coords = (w // 2, int(h * 0.93))
        self.reset()

    def reset(self) -> None:
        self.screen = DonationScreen.VILLAGE
        self.request_index = 0
        self.troop_selected = False
        self.donated = 0
        self.done = False
        self.own_resources = self._copy_resources(self._initial_resources)

    @property
    def current_request(self) -> DonationRequest | None:
        if 0 <= self.request_index < len(self.requests):
            return self.requests[self.request_index]
        return None

    # ------------------------------------------------------------------ 感知
    def perceive(self) -> DonationSignals:
        if self.screen is DonationScreen.VILLAGE:
            if self.done:
                return DonationSignals()
            return DonationSignals(message_list_button=self.message_list_button)
        if self.screen is DonationScreen.MESSAGE_LIST:
            request = self.current_request
            return DonationSignals(
                request_entry=request.coords if request is not None else None,
                close_button=self.close_button,
                troop_type=request.troop_type if request else None,
                request_cost=request.cost if request else 0,
                request_resource_type=request.resource_type if request else "elixir",
                resources=self.own_resources,
            )
        if self.screen is DonationScreen.REQUEST_DETAIL:
            return DonationSignals(
                reinforce_button=self.reinforce_button,
                close_button=self.close_button,
                troop_type=self.current_request.troop_type if self.current_request else None,
            )
        if self.screen is DonationScreen.DONATING:
            return DonationSignals(
                troop_bar_coords=self.troop_bar_coords,
                confirm_button=self.confirm_button,
                troop_type=self.current_request.troop_type if self.current_request else None,
            )
        return DonationSignals(
            close_button=self.close_button,
            more_requests=self.request_index + 1 < len(self.requests),
        )

    # ----------------------------------------------------------------- 执行
    def step(self, action: Any) -> SimulationResult:
        kind = getattr(action, "kind", None)
        target = getattr(action, "target", None)
        coords = getattr(action, "coords", None)
        if kind == "tap" and coords is not None:
            return self._tap(target, tuple(coords))
        if kind == "wait":
            return SimulationResult("wait", 0.0, self.done, {"reason": "等待识别", "screen_changed": False})
        if kind == "stop":
            self.done = True
            return SimulationResult("stop", 0.0, True, {"reason": "捐兵流程停止", "screen_changed": False})
        return SimulationResult(kind or "wait", 0.0, self.done, {"reason": "无效动作", "screen_changed": False})

    def _tap(self, target: str | None, coords: tuple[int, int]) -> SimulationResult:
        if self.screen is DonationScreen.VILLAGE:
            if target == "消息列表按钮" and self._near(coords, self.message_list_button):
                self.screen = DonationScreen.MESSAGE_LIST
                return SimulationResult("tap", 0.0, False, {"reason": "进入消息列表", "screen_changed": True})

        if self.screen is DonationScreen.MESSAGE_LIST:
            request = self.current_request
            if target == "请求条目" and request is not None and self._near(coords, request.coords):
                self.screen = DonationScreen.REQUEST_DETAIL
                return SimulationResult("tap", 0.0, False, {"reason": "进入请求详情", "screen_changed": True})
            if target == "关闭按钮" and self._near(coords, self.close_button):
                self.screen = DonationScreen.VILLAGE
                return SimulationResult("tap", 0.0, False, {"reason": "关闭消息列表", "screen_changed": True})

        if self.screen is DonationScreen.REQUEST_DETAIL:
            if target == "增援按钮" and self._near(coords, self.reinforce_button):
                self.screen = DonationScreen.DONATING
                self.troop_selected = False
                return SimulationResult("tap", 0.0, False, {"reason": "点击增援", "screen_changed": True})
            if target == "关闭按钮" and self._near(coords, self.close_button):
                self.screen = DonationScreen.MESSAGE_LIST
                return SimulationResult("tap", 0.0, False, {"reason": "取消增援返回列表", "screen_changed": True})

        if self.screen is DonationScreen.DONATING:
            if target == "兵种选择栏" and self._near(coords, self.troop_bar_coords):
                self.troop_selected = True
                return SimulationResult("tap", 0.0, False, {"reason": "选中兵种", "screen_changed": False})
            if (
                target == "捐赠确认按钮"
                and self.troop_selected
                and self._near(coords, self.confirm_button)
            ):
                request = self.current_request
                if request is not None and not self._affordable(request):
                    return SimulationResult(
                        "tap",
                        0.0,
                        self.done,
                        {"reason": "己方资源不足以支付捐款", "screen_changed": False},
                    )
                self.donated += 1
                if request is not None:
                    self._spend(request)
                self.screen = DonationScreen.DONE
                return SimulationResult("tap", 0.0, False, {"reason": "捐赠完成", "screen_changed": True})

        if self.screen is DonationScreen.DONE:
            if target == "关闭按钮" and self._near(coords, self.close_button):
                if self.request_index + 1 < len(self.requests):
                    self.request_index += 1
                    self.screen = DonationScreen.MESSAGE_LIST
                else:
                    self.screen = DonationScreen.VILLAGE
                    self.done = True
                return SimulationResult("tap", 0.0, self.done, {"reason": "关闭完成返回", "screen_changed": True})

        return SimulationResult("tap", 0.0, self.done, {"reason": f"无效点击 {target}", "screen_changed": False})

    def _near(self, a: tuple[int, int], b: tuple[int, int], tol: float = 12.0) -> bool:
        return math.dist(a, b) <= tol

    @staticmethod
    def _copy_resources(resources: Resources) -> Resources:
        return Resources(
            gold=int(resources.gold),
            elixir=int(resources.elixir),
            dark_elixir=int(resources.dark_elixir),
        )

    def _affordable(self, request: DonationRequest) -> bool:
        if request.cost <= 0:
            return True
        return getattr(self.own_resources, request.resource_type, 0) >= request.cost

    def _spend(self, request: DonationRequest) -> None:
        if request.cost <= 0:
            return
        current = getattr(self.own_resources, request.resource_type, 0)
        setattr(self.own_resources, request.resource_type, max(0, current - request.cost))

    def summary(self) -> dict:
        return {
            "requests": len(self.requests),
            "donated": self.donated,
            "screen": self.screen.value,
            "done": self.done,
            "resources": {
                "gold": self.own_resources.gold,
                "elixir": self.own_resources.elixir,
                "dark_elixir": self.own_resources.dark_elixir,
            },
        }
