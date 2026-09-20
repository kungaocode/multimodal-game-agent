"""Structured action execution for coach sessions.

Handles tap and deploy actions on a real device (ADB) or emulator.
Deploy is a two-step action: tap the troop bar to select a troop type,
then tap the landing point on the battlefield. The landing point is
planned by tools.coach_masks.plan_deploy_point to avoid the red zone.
"""

from __future__ import annotations

import time
from typing import Any

from PIL import Image

from tools.coach_masks import plan_deploy_point


def execute_action(
    action_dict: dict,
    adb: object | None,
    fsm_farm: Any | None,
    fsm_donation: Any | None,
    module: str,
    image_size: tuple[int, int] | None = None,
    screenshot: Image.Image | None = None,
    detections: list[dict] | None = None,
) -> dict:
    """Execute a structured action. Returns an execution result dict.

    deploy = real two-step deploy: tap the troop bar first (coach-provided
    troop_coords preferred, fallback to bottom-center default), then tap the
    planned landing point. The landing point is planned to land on grass
    outside the red zone, adjacent to the target building.
    """
    kind = action_dict.get("kind", "wait")
    coords = action_dict.get("coords")
    if kind in ("wait", "stop") or coords is None or adb is None:
        return {"executed": False, "reason": f"skip {kind}"}
    try:
        if kind == "tap":
            adb.tap(coords[0], coords[1])
            return {"executed": True, "action": "tap", "coords": list(coords)}
        if kind == "deploy":
            w, h = screenshot.size if screenshot is not None else (image_size or (1280, 960))
            troop_coords = action_dict.get("troop_coords")
            if troop_coords is not None and len(troop_coords) >= 2:
                bar_x, bar_y = int(troop_coords[0]), int(troop_coords[1])
                troop_source = "coach"
            else:
                bar_x, bar_y = int(w * 0.5), int(h * 0.92)
                troop_source = "default"
            planned = bool(action_dict.get("landing_planned"))
            if screenshot is not None and not planned:
                pts = list(
                    plan_deploy_point(
                        coords,
                        action_dict.get("target"),
                        detections,
                        screenshot,
                    )
                )
            else:
                pts = list(coords)
            adb.tap(bar_x, bar_y)
            time.sleep(0.4)
            adb.tap(pts[0], pts[1])
            return {
                "executed": True,
                "action": "deploy",
                "troop_bar": [bar_x, bar_y],
                "troop_source": troop_source,
                "coords": pts,
                "raw_coords": list(coords),
                "landing_plan": "building_adjacent",
            }
    except Exception as exc:
        return {"executed": False, "error": str(exc)}
    return {"executed": False, "reason": f"unknown kind {kind}"}
