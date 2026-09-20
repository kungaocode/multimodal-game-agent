"""Coach session: local model -> coach review -> execute -> record.

This module is the thin orchestrator for the "teacher-student" loop. Image
masks live in ``coach_masks``, execution in ``coach_executor`` and training
data storage in ``coach_recorder``.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from PIL import Image

from decision.battle_fsm import BattleFSM, BattleSignals
from decision.coach import CoachReviewer, CoachVerdict
from decision.deploy_planner import DeployPlanner
from decision.donation_fsm import DonationFSM, DonationSignals
from decision.farm_fsm import FarmSignals
from decision.resource_policy import ResourcePolicy
from executor.adb_executor import AdbExecutor
from executor.action import AgentAction
from executor.validator import ActionValidator
from perception.fsm_labels import EXTENDED_FSM_ORDER, EXTENDED_NAME_TO_ID

# Backward-compatible names used by API callers and tests.
from tools.coach_executor import execute_action as _execute_action
from tools.coach_masks import (
    green_mask as _green_mask,
    no_land_mask as _no_land_mask,
    plan_deploy_point as _plan_deploy_point,
    red_zone_mask as _red_zone_mask,
    snap_deploy_point as _snap_deploy_point,
)
from tools.coach_recorder import TrainingDataRecorder

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
LOCAL_ADB_PATH = ROOT / "tools/android-platform-tools/platform-tools/adb"


def _adb_path() -> str:
    if LOCAL_ADB_PATH.is_file() and os.access(LOCAL_ADB_PATH, os.X_OK):
        return str(LOCAL_ADB_PATH)
    return "adb"


def _center(det: dict) -> tuple[int, int] | None:
    if det.get("coords"):
        return int(det["coords"][0]), int(det["coords"][1])
    bbox = det.get("bbox")
    if bbox and len(bbox) >= 4:
        return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
    return None


def _validated_action(action: dict, screenshot: Image.Image) -> dict:
    """Keep real-device execution inside the validator's safety envelope."""
    validator = ActionValidator(
        image_size=screenshot.size,
        allowed_targets=EXTENDED_FSM_ORDER,
    )
    check = validator.validate(AgentAction.from_dict(action))
    if check.allowed:
        return action
    logger.warning("Action rejected: %s", check.reason)
    return {"kind": "wait", "target": check.reason, "coords": None, "troop_coords": None}


def _local_perceive_and_decide(
    screenshot: Image.Image,
    detector,
    fsm_farm: BattleFSM | None,
    fsm_donation: DonationFSM | None,
    module: str,
) -> tuple[str, dict, list[dict], BattleSignals | DonationSignals]:
    """Run local perception + FSM decision.

    Returns ``(state_name, action_dict, detections, signals)``.
    """
    detections: list[dict] = []
    if detector is not None:
        try:
            detections = [
                {
                    "class_id": int(d.class_id),
                    "type": d.class_name,
                    "bbox": list(d.bbox),
                    "confidence": float(d.confidence),
                }
                for d in detector.predict(screenshot)
            ]
        except Exception as exc:
            logger.warning("Local detection failed: %s", exc)

    by_type: dict[str, list[dict]] = {}
    for det in detections:
        by_type.setdefault(det["type"], []).append(det)

    def first(name: str) -> tuple[int, int] | None:
        centers = [center for det in by_type.get(name, []) if (center := _center(det))]
        return centers[0] if centers else None

    if module == "donation":
        signals = DonationSignals(
            message_list_button=first("消息列表按钮"),
            request_entry=first("请求条目"),
            reinforce_button=first("增援按钮"),
            close_button=first("关闭按钮"),
            confirm_button=first("捐赠确认按钮"),
        )
        assert fsm_donation is not None
        action = fsm_donation.step(signals)
        state_name = fsm_donation.state.value
    else:
        enemy_resources: list[tuple[str, tuple[int, int]]] = []
        for name in (
            "圣水收集器",
            "金矿",
            "圣水瓶",
            "储金罐",
            "暗黑重油罐",
            "暗黑重油钻井",
        ):
            for det in by_type.get(name, []):
                center = _center(det)
                if center is not None:
                    enemy_resources.append((name, center))
        signals = BattleSignals(
            attack_button=first("进攻按钮"),
            search_button=first("搜索对手按钮"),
            next_button=first("下一个按钮"),
            return_button=first("返回按钮"),
            end_battle_button=first("结束战斗按钮"),
            enemy_resources=enemy_resources,
        )
        assert fsm_farm is not None
        action = fsm_farm.step(signals)
        state_name = fsm_farm.state.value

    action_dict = {
        "kind": action.kind,
        "target": action.target,
        "coords": list(action.coords) if action.coords is not None else None,
        "troop_coords": None,
    }
    return state_name, action_dict, detections, signals


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Coach session: local model + Qwen-VL review + execute"
    )
    parser.add_argument("--module", choices=["farm", "donation"], default="farm")
    parser.add_argument("--serial", help="ADB device serial")
    parser.add_argument("--video", help="Video file (dry-run mode, no device)")
    parser.add_argument("--weights", help="YOLO weights path")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--step-delay", type=float, default=1.0)
    parser.add_argument("--out", default="dataset/coach_sessions")
    parser.add_argument("--no-coach", action="store_true", help="Disable coach review")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights) if args.weights else None
    if weights is None:
        candidates = [
            Path("runs/detect/fsm_v4/weights/best.pt"),
            Path("runs/detect/fsm_v2/weights/best.pt"),
            Path("runs/detect/fsm/weights/best.pt"),
        ]
        weights = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    detector = None
    if weights.exists():
        try:
            from perception.detector import Detector
            detector = Detector(weights)
            logger.info("Detector loaded: %s", weights)
        except Exception as exc:
            logger.warning("Detector unavailable: %s", exc)
    else:
        logger.warning("Weights not found at %s, running without local detector.", weights)

    vision = None
    coach = None
    if not args.no_coach:
        try:
            from perception.vision_model import VisionModel
            vision = VisionModel()
            if vision.base_url:
                coach = CoachReviewer(vision)
                logger.info("Coach enabled: %s", vision.model)
            else:
                logger.warning("VISION_MODEL_BASE_URL not set, coach disabled.")
        except Exception as exc:
            logger.warning("Coach setup failed: %s", exc)

    if args.video:
        from perception.auto_labeler import extract_frames
        video_frames = extract_frames(args.video, max_frames=args.max_steps)
        logger.info("Dry-run mode: %d frames from %s", len(video_frames), args.video)
        adb = None
    else:
        video_frames = None
        adb = AdbExecutor(serial=args.serial, adb_path=_adb_path())
        if not adb.is_connected():
            raise RuntimeError("No ADB device connected. Use --serial or connect device.")
        logger.info("ADB connected: %s", ", ".join(adb.devices()))

    policy = ResourcePolicy()
    deploy_planner = DeployPlanner(
        policy=policy,
        mask_provider=lambda _signals, screenshot: _no_land_mask(screenshot) if screenshot else None,
        landing_planner=lambda name, coords, shot, dets, requested: (
            _plan_deploy_point(requested or coords, name, dets, shot) if shot else coords
        ),
        max_consecutive_per_target=2,
    )
    fsm_farm = BattleFSM(policy=policy) if args.module == "farm" else None
    fsm_donation = DonationFSM() if args.module == "donation" else None
    recorder = TrainingDataRecorder(out_dir)
    logger.info("Coach session started: module=%s, output=%s", args.module, out_dir)

    for step in range(1, args.max_steps + 1):
        try:
            screenshot = video_frames[step - 1] if video_frames is not None else adb.screencap()
        except Exception as exc:
            logger.error("Screencap failed at step %d: %s", step, exc)
            break

        local_state, local_action, detections, signals = _local_perceive_and_decide(
            screenshot, detector, fsm_farm, fsm_donation, args.module
        )
        logger.info(
            "Step %d: state=%s, action=%s, detections=%d",
            step,
            local_state,
            local_action,
            len(detections),
        )

        verdict: CoachVerdict | None = None
        if coach is not None:
            try:
                verdict = coach.review(
                    screenshot=screenshot,
                    local_state=local_state,
                    local_action_kind=local_action.get("kind"),
                    local_action_target=local_action.get("target"),
                    local_action_coords=tuple(local_action["coords"]) if local_action.get("coords") else None,
                    local_detections=detections,
                    module=args.module,
                    fsm_history=fsm_farm.history if fsm_farm else fsm_donation.history,
                )
                logger.info(
                    "  Coach: approved=%s, corrected_state=%s, reasoning=%.60s",
                    verdict.approved,
                    verdict.corrected_state,
                    verdict.reasoning,
                )
            except Exception as exc:
                logger.warning("Coach error at step %d: %s", step, exc)

        final_action = local_action
        if verdict is not None and not verdict.approved and verdict.corrected_action_kind:
            final_action = {
                "kind": verdict.corrected_action_kind,
                "target": verdict.corrected_action_target,
                "coords": (
                    list(verdict.corrected_action_coords)
                    if verdict.corrected_action_coords
                    else None
                ),
                "troop_coords": (
                    list(verdict.corrected_troop_coords)
                    if verdict.corrected_troop_coords
                    else None
                ),
            }
            logger.info(
                "  Coach corrected: %s -> %s",
                local_action.get("kind"),
                final_action.get("kind"),
            )
        if verdict is not None and not verdict.approved and verdict.corrected_state:
            local_state = verdict.corrected_state

        deploy_plan = None
        if final_action.get("kind") == "deploy":
            deploy_plan = deploy_planner.plan(
                signals,
                screenshot=screenshot,
                detections=detections,
                requested_name=final_action.get("target"),
                requested_coords=tuple(final_action["coords"]) if final_action.get("coords") else None,
            )
            if deploy_plan is None:
                logger.info("  Deploy rejected: no scoreable target")
            else:
                final_action = {
                    "kind": "deploy",
                    "target": deploy_plan.target_name,
                    "coords": list(deploy_plan.landing_point),
                    "troop_coords": final_action.get("troop_coords"),
                    "target_score": deploy_plan.score,
                    "target_breakdown": deploy_plan.breakdown,
                    "target_repeat_count": deploy_plan.repeat_count,
                    "target_reason": deploy_plan.reason,
                    "landing_planned": True,
                }
                logger.info(
                    "  Deploy plan: %s score=%.3f repeat=%d reason=%s",
                    deploy_plan.target_name,
                    deploy_plan.score,
                    deploy_plan.repeat_count,
                    deploy_plan.reason,
                )
        elif local_state == "战斗结束":
            deploy_planner.reset()

        final_action = _validated_action(final_action, screenshot)
        exec_result = _execute_action(
            final_action,
            adb,
            fsm_farm,
            fsm_donation,
            args.module,
            image_size=screenshot.size,
            screenshot=screenshot,
            detections=detections,
        )
        recorder.record(
            step=step,
            screenshot=screenshot,
            local_state=local_state,
            local_action=local_action,
            local_detections=detections,
            coach_verdict=verdict,
            executed=bool(exec_result.get("executed", False)),
            result=exec_result,
        )

        if final_action.get("kind") == "stop":
            logger.info("FSM reached stop at step %d.", step)
            break
        if video_frames is None and args.step_delay > 0:
            time.sleep(args.step_delay)

    summary = recorder.save_summary(args.module, args.max_steps)
    print("===== Coach session summary =====")
    print(f"Module:            {summary['module']}")
    print(f"Total steps:       {summary['total_steps']}")
    print(f"Coach approved:    {summary['coach_approved']}")
    print(f"Coach corrected:   {summary['coach_corrected']}")
    print(f"Correction rate:   {summary['coach_correction_rate']:.1%}")
    print(f"Executed actions:  {summary['executed_actions']}")
    print(f"Output:            {out_dir}")
    print("Next: python -m tools.coach_retrain --sessions", args.out)


if __name__ == "__main__":
    main()
