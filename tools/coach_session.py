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
from dataclasses import replace
from pathlib import Path

from PIL import Image

from decision.battle_fsm import BattleFSM, BattleSignals, BattleState, EmptyTargetGuard
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


# 护栏识别为「结束/返回」类动作的目标名：这类动作必须先过 deploy_planner 判空
_END_ACTION_TARGETS = ("结束战斗", "返回按钮")


def _deploy_action_from_plan(plan, troop_coords=None) -> dict:
    """由 DeployPlan 构建完整 deploy 动作 dict（含打分/重复次数元数据，便于日志与审计）。"""
    return {
        "kind": "deploy",
        "target": plan.target_name,
        "coords": list(plan.landing_point),
        "troop_coords": list(troop_coords) if troop_coords else None,
        "target_score": plan.score,
        "target_breakdown": plan.breakdown,
        "target_repeat_count": plan.repeat_count,
        "target_reason": plan.reason,
        "landing_planned": True,
    }


def _backfill_resources(signals, verdict):
    """把教练枚举的可见资源建筑并入 signals.enemy_resources（按 (名, 坐标) 去重）。

    这样 deploy_planner.plan(...) 打分/判空时能吃到「全部可见资源建筑」，
    而不是只有本地检测器（真机常 0 检出）给出的单个目标。
    """
    if verdict is None or not getattr(verdict, "resource_buildings", None):
        return signals
    existing = set(signals.enemy_resources)
    added = [
        (name, coords)
        for name, coords in verdict.resource_buildings
        if (name, coords) not in existing
    ]
    if not added:
        return signals
    return replace(signals, enemy_resources=[*signals.enemy_resources, *added])


def _battle_end_guard(
    signals,
    deploy_planner,
    final_action: dict,
    guard: EmptyTargetGuard,
    screenshot=None,
    detections: list[dict] | None = None,
) -> tuple[dict, object | None]:
    """战斗内过早结束护栏：以确定性资源算法(deploy_planner.plan)为结束判定唯一权威。

    - 本帧已决定下兵 → 记为有目标（清空判空计数），原样放行；
    - 本帧拟 tap 结束/返回 → 用回填后的 signals 再问一遍算法：
        仍有可获取目标 → 强制改下兵并清空判空计数；
        无可获取目标但连续判空未达阈值 → 改 wait（暂不结束）；
        达阈值 → 放行原结束动作。
    返回 (可能被改写的 final_action, 已规划的 deploy_plan 或 None)。
    """
    kind = final_action.get("kind")
    target = final_action.get("target")
    if kind == "deploy":
        guard.on_targets()
        return final_action, None
    if kind == "tap" and target in _END_ACTION_TARGETS:
        plan = deploy_planner.plan(signals, screenshot=screenshot, detections=detections)
        if plan is not None:
            guard.on_targets()
            return _deploy_action_from_plan(
                plan, troop_coords=final_action.get("troop_coords")
            ), plan
        if not guard.on_empty():
            logger.info(
                "  End guard: 判空 %d/%d，无可获取目标，暂不点结束",
                guard.streak,
                guard.threshold,
            )
            return (
                {
                    "kind": "wait",
                    "target": "判空未达阈值，暂不结束战斗",
                    "coords": None,
                    "troop_coords": None,
                },
                None,
            )
    return final_action, None


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
    parser.add_argument(
        "--end-empty-frames",
        type=int,
        default=3,
        help="战斗内连续多少帧无可获取目标才放行点结束战斗（默认 3）",
    )
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
    fsm_farm = (
        BattleFSM(policy=policy, empty_frames_threshold=max(1, args.end_empty_frames))
        if args.module == "farm"
        else None
    )
    fsm_donation = DonationFSM() if args.module == "donation" else None
    # 与 BattleFSM 共享同一判空守卫：本地 FSM 的结束判定与 coach_session 护栏不两处分叉
    guard = (
        fsm_farm.empty_guard
        if fsm_farm is not None
        else EmptyTargetGuard(threshold=max(1, args.end_empty_frames))
    )
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
        if args.module == "farm":
            # (a) 资源回填：教练枚举的可见资源建筑并入 signals（真机 0 检出时，
            #     这是算法能看到的全部资源，决定「打不打光/何时可结束」）。
            signals = _backfill_resources(signals, verdict)
            # (b) 战斗内过早结束护栏：只有算法连续判空达阈值才放行「结束/返回」。
            if fsm_farm is not None and fsm_farm.state is BattleState.BATTLE:
                final_action, deploy_plan = _battle_end_guard(
                    signals,
                    deploy_planner,
                    final_action,
                    guard,
                    screenshot=screenshot,
                    detections=detections,
                )
                if deploy_plan is not None:
                    logger.info(
                        "  End guard: 算法仍有可获取目标，改下兵 %s (score=%.3f)",
                        deploy_plan.target_name,
                        deploy_plan.score,
                    )

        if final_action.get("kind") == "deploy":
            if deploy_plan is None:
                deploy_plan = deploy_planner.plan(
                    signals,
                    screenshot=screenshot,
                    detections=detections,
                    requested_name=final_action.get("target"),
                    requested_coords=tuple(final_action["coords"]) if final_action.get("coords") else None,
                )
            if deploy_plan is None:
                logger.info("  Deploy rejected: no scoreable target")
                final_action = {
                    "kind": "wait",
                    "target": "无可打分目标，等待下一帧",
                    "coords": None,
                    "troop_coords": None,
                }
            else:
                final_action = _deploy_action_from_plan(
                    deploy_plan, troop_coords=final_action.get("troop_coords")
                )
                logger.info(
                    "  Deploy plan: %s score=%.3f repeat=%d reason=%s",
                    deploy_plan.target_name,
                    deploy_plan.score,
                    deploy_plan.repeat_count,
                    deploy_plan.reason,
                )
        elif local_state == "战斗结束":
            guard.reset()
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
