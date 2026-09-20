"""Coach-guided session lifecycle for mobile/HarmonyOS clients.

The backend owns model loading, ADB execution, coach review and retraining.
A phone client only sends a start request and observes progress.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, Field

from decision.coach import CoachReviewer
from decision.donation_fsm import DonationFSM
from decision.battle_fsm import BattleFSM
from executor.adb_executor import AdbExecutor
from executor.action import ALLOWED_ACTION_KINDS
from executor.validator import ActionValidator
from perception.fsm_labels import EXTENDED_FSM_ORDER
from tools.coach_retrain import extract_state_labels, extract_yolo_data
from tools.coach_executor import execute_action as _execute_action
from tools.coach_recorder import TrainingDataRecorder
from tools.coach_session import _local_perceive_and_decide, _validated_action

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = ROOT / "dataset" / "video"
SESSION_ROOT = ROOT / "dataset" / "coach_sessions"
DETECTOR_V1 = Path(os.getenv("DETECTOR_MODEL", ROOT / "runs/detect/fsm_v4/weights/best.pt"))
STATE_V1 = Path(os.getenv("STATE_CLASSIFIER_MODEL", ROOT / "runs/state_classifier_v4/best.pt"))
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MODULE_DIR_NAMES = {"farm": "resource", "donation": "donate"}
LOCAL_ADB_PATH = ROOT / "tools/android-platform-tools/platform-tools/adb"


def _adb_path() -> str:
    if LOCAL_ADB_PATH.is_file() and os.access(LOCAL_ADB_PATH, os.X_OK):
        return str(LOCAL_ADB_PATH)
    return "adb"


class SessionStartRequest(BaseModel):
    module: Literal["farm", "donation"] = "farm"
    source: Literal["adb", "video"] = "video"
    serial: str | None = Field(default=None, description="ADB device serial")
    video_path: str | None = Field(default=None, description="Relative path under dataset/video")
    max_steps: int = Field(default=20, ge=1, le=500)
    step_delay: float = Field(default=0.5, ge=0, le=60)
    video_fps: float = Field(default=0.5, ge=0.05, le=10)
    enable_coach: bool = True
    weights: str | None = None


class RetrainRequest(BaseModel):
    sessions_dir: Path = Path("dataset/coach_sessions")
    out: Path = Path("dataset/coach_training")
    detector_out: Path = Path("runs/detect/fsm_v5")
    state_out: Path = Path("runs/state_classifier_v5")
    base_weights: Path = Path("yolov8n.pt")
    yolo_epochs: int = Field(default=5, ge=1, le=100)
    state_epochs: int = Field(default=100, ge=10, le=1000)
    imgsz: int = Field(default=512, ge=320, le=1024)
    batch: int = Field(default=8, ge=1, le=64)
    train_detector: bool = True
    train_state: bool = True


class SessionManager:
    """Small in-memory job manager. Restarting the server clears live state."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._retrain_jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> dict[str, Any]:
        detector_configured = DETECTOR_V1.exists()
        state_configured = STATE_V1.exists()
        coach_configured = bool(os.getenv("VISION_MODEL_BASE_URL") and os.getenv("VISION_MODEL_API_KEY"))
        adb_path = _adb_path()
        adb_ok = adb_path != "adb" or shutil.which("adb") is not None
        return {
            "modules": [
                {"id": "farm", "name": "打资源", "default_video_dir": "dataset/video/resource"},
                {"id": "donation", "name": "捐兵", "default_video_dir": "dataset/video/donate"},
            ],
            "models": {
                "detector_v1": str(DETECTOR_V1) if detector_configured else None,
                "state_classifier_v1": str(STATE_V1) if state_configured else None,
                "coach_configured": coach_configured,
            },
            "execution": {"adb_cli": adb_ok, "video_dry_run": True},
            "training": {
                "detector": detector_configured,
                "state_classifier": state_configured,
            },
        }

    def list_videos(self, module: str) -> list[str]:
        directory = VIDEO_ROOT / MODULE_DIR_NAMES[module]
        if not directory.exists():
            return []
        return sorted(
            str(path.relative_to(ROOT)) for path in directory.rglob("*")
            if path.suffix.lower() in VIDEO_SUFFIXES
        )

    def save_video(self, module: str, filename: str, data: bytes) -> str:
        if module not in {"farm", "donation"}:
            raise ValueError("module 只能是 farm 或 donation")
        if not data:
            raise ValueError("上传文件为空")
        if Path(filename).suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError("只支持 mp4/mov/mkv/webm/avi")
        safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(filename).name)
        target_dir = VIDEO_ROOT / MODULE_DIR_NAMES[module]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}"
        target.write_bytes(data)
        return str(target.relative_to(ROOT))

    def start(self, request: SessionStartRequest) -> dict[str, Any]:
        session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        out_dir = SESSION_ROOT / session_id
        out_dir.mkdir(parents=True, exist_ok=True)

        video_path: Path | None = None
        if request.source == "video":
            if not request.video_path:
                raise ValueError("视频运行模式需要 video_path")
            candidate = Path(request.video_path)
            if not candidate.is_absolute():
                candidate = ROOT / candidate
            candidate = candidate.resolve()
            if ROOT not in candidate.parents and candidate != ROOT:
                raise ValueError("视频路径必须在项目目录内")
            if not candidate.is_file() or candidate.suffix.lower() not in VIDEO_SUFFIXES:
                raise ValueError(f"视频不存在或格式不支持: {request.video_path}")
            video_path = candidate

        session: dict[str, Any] = {
            "session_id": session_id,
            "request": request.model_dump(),
            "output_dir": str(out_dir.relative_to(ROOT)),
            "status": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "current_step": 0,
            "total_steps": request.max_steps,
            "steps": [],
            "summary": None,
            "stop_requested": False,
        }
        with self._lock:
            self._sessions[session_id] = session
        thread = threading.Thread(
            target=self._run_session,
            args=(session_id, request, out_dir, video_path),
            name=f"coach-session-{session_id}",
            daemon=True,
        )
        thread.start()
        return self._public(session)

    def get(self, session_id: str, steps_limit: int = 30) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            public = self._public(session)
        public["steps"] = public["steps"][-steps_limit:]
        return public

    def request_stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if session["status"] in ("queued", "running"):
                session["stop_requested"] = True
                session["status"] = "stopping"
            return self._public(session)

    def _public(self, session: dict[str, Any]) -> dict[str, Any]:
        return {
            **session,
            "stop_requested": session["stop_requested"],
        }

    def _update(self, session_id: str, **fields: Any) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.update(fields)

    def _append_step(self, session_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session["steps"].append(record)
            session["current_step"] = record["step"]

    def _run_session(
        self,
        session_id: str,
        request: SessionStartRequest,
        out_dir: Path,
        video_path: Path | None,
    ) -> None:
        self._update(session_id, status="running", started_at=time.time())
        try:
            detector = None
            weights = request.weights or str(DETECTOR_V1)
            if Path(weights).is_file():
                from perception.detector import Detector
                detector = Detector(weights)

            classifier = None
            if STATE_V1.is_file():
                from decision.state_classifier import StateClassifier
                classifier = StateClassifier.load(STATE_V1)

            coach = None
            if request.enable_coach:
                from perception.vision_model import VisionModel
                vision = VisionModel()
                if vision.base_url:
                    coach = CoachReviewer(vision)

            adb = None
            frames: list[Image.Image] | None = None
            if video_path is not None:
                from perception.auto_labeler import extract_frames
                frames = extract_frames(video_path, fps=request.video_fps, max_frames=request.max_steps)
                total_steps = min(request.max_steps, len(frames))
            else:
                adb = AdbExecutor(serial=request.serial, adb_path=_adb_path())
                if not adb.is_connected():
                    raise RuntimeError("ADB 设备未连接")
                total_steps = request.max_steps

            self._update(session_id, total_steps=total_steps)
            fsm_farm = BattleFSM() if request.module == "farm" else None
            fsm_donation = DonationFSM() if request.module == "donation" else None
            recorder = TrainingDataRecorder(out_dir)

            for step in range(1, total_steps + 1):
                with self._lock:
                    stop_requested = bool(self._sessions[session_id]["stop_requested"])
                if stop_requested:
                    break
                screenshot = frames[step - 1] if frames is not None else adb.screencap()
                local_state, local_action, detections, signals = _local_perceive_and_decide(
                    screenshot, detector, fsm_farm, fsm_donation, request.module
                )
                validator = ActionValidator(
                    image_size=screenshot.size,
                    allowed_kinds=ALLOWED_ACTION_KINDS,
                    allowed_targets=EXTENDED_FSM_ORDER,
                )
                check = validator.validate(_to_agent_action(local_action))
                if not check.allowed:
                    local_action = {"kind": "wait", "target": check.reason, "coords": None}

                verdict = None
                if coach is not None:
                    verdict = coach.review(
                        screenshot=screenshot,
                        local_state=local_state,
                        local_action_kind=local_action.get("kind"),
                        local_action_target=local_action.get("target"),
                        local_action_coords=tuple(local_action["coords"]) if local_action.get("coords") else None,
                        local_detections=detections,
                        module=request.module,
                        fsm_history=fsm_farm.history if fsm_farm else fsm_donation.history,
                    )

                final_action = local_action
                if verdict and not verdict.approved and verdict.corrected_action_kind:
                    final_action = {
                        "kind": verdict.corrected_action_kind,
                        "target": verdict.corrected_action_target,
                        "coords": list(verdict.corrected_action_coords) if verdict.corrected_action_coords else None,
                        "troop_coords": list(verdict.corrected_troop_coords) if verdict.corrected_troop_coords else None,
                    }
                if verdict and not verdict.approved and verdict.corrected_state:
                    local_state = verdict.corrected_state

                final_action = _validated_action(final_action, screenshot)
                result = _execute_action(
                    final_action,
                    adb,
                    fsm_farm,
                    fsm_donation,
                    request.module,
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
                    executed=bool(result.get("executed", False)),
                    result=result,
                )
                step_record = {
                    "step": step,
                    "local_state": local_state,
                    "classifier_state": classifier.predict(detections) if classifier else None,
                    "action": final_action,
                    "coach_approved": verdict.approved if verdict else None,
                    "coach_correction": verdict.corrected_state if verdict and not verdict.approved else None,
                    "reasoning": verdict.reasoning if verdict else None,
                    "executed": result,
                }
                self._append_step(session_id, step_record)
                if final_action.get("kind") == "stop":
                    break
                if request.step_delay > 0 and step < total_steps:
                    time.sleep(request.step_delay)

            summary = recorder.save_summary(request.module, request.max_steps)
            with self._lock:
                session = self._sessions[session_id]
                session["summary"] = summary
                session["status"] = "stopping" if session["stop_requested"] else "completed"
                session["finished_at"] = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Coach session failed: %s", session_id)
            self._update(session_id, status="failed", error=str(exc), finished_at=time.time())

    def start_retrain(self, request: RetrainRequest) -> dict[str, Any]:
        job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        sessions_dir = _safe_path(request.sessions_dir)
        if not sessions_dir.is_dir():
            raise ValueError(f"会话目录不存在: {request.sessions_dir}")
        job = {
            "job_id": job_id,
            "status": "queued",
            "sessions_dir": str(sessions_dir.relative_to(ROOT)),
            "out": str(_safe_path(request.out).relative_to(ROOT)),
            "logs": [],
            "error": None,
            "created_at": time.time(),
        }
        with self._lock:
            self._retrain_jobs[job_id] = job
        thread = threading.Thread(
            target=self._run_retrain,
            args=(job_id, request),
            name=f"coach-retrain-{job_id}",
            daemon=True,
        )
        thread.start()
        return job

    def retrain_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._retrain_jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def _log(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._retrain_jobs.get(job_id)
            if job is not None:
                job["logs"].append(message)

    def _run_retrain(self, job_id: str, request: RetrainRequest) -> None:
        self._log(job_id, "开始提取回流数据")
        self._update_job(job_id, status="running")
        try:
            sessions_dir = _safe_path(request.sessions_dir)
            out_dir = _safe_path(request.out)
            session_dirs = [
                path for path in sorted(sessions_dir.iterdir())
                if path.is_dir() and (path / "session_summary.json").is_file()
            ]
            if not session_dirs:
                raise ValueError("没有包含 session_summary.json 的教练会话")
            out_dir.mkdir(parents=True, exist_ok=True)
            yolo_count = extract_yolo_data(session_dirs, out_dir)
            state_count = extract_state_labels(session_dirs, out_dir)
            self._log(job_id, f"YOLO 样本 {yolo_count}；状态样本 {state_count}")

            if request.train_detector:
                self._log(job_id, "开始 YOLO 微调")
                subprocess.run(
                    [
                        sys.executable, "-m", "tools.train_yolo",
                        "--dataset", str(out_dir / "yolo"),
                        "--base", str(request.base_weights),
                        "--out", str(request.detector_out),
                        "--epochs", str(request.yolo_epochs),
                        "--imgsz", str(request.imgsz),
                        "--batch", str(request.batch),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60 * 60 * 6,
                )
                self._log(job_id, f"YOLO 完成: {request.detector_out}")

            if request.train_state:
                self._log(job_id, "开始状态分类器训练")
                subprocess.run(
                    [
                        sys.executable, "-m", "decision.state_classifier",
                        "--train",
                        "--data", str(out_dir / "state_labels.json"),
                        "--out", str(request.state_out),
                        "--epochs", str(request.state_epochs),
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60 * 60 * 2,
                )
                self._log(job_id, f"状态分类器完成: {request.state_out}")
            self._update_job(job_id, status="completed", error=None)
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "")[-4000:]
            self._log(job_id, f"训练命令失败: {stderr}")
            self._update_job(job_id, status="failed", error=stderr or "训练命令失败")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Retrain failed: %s", job_id)
            self._update_job(job_id, status="failed", error=str(exc))

    def _update_job(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._retrain_jobs.get(job_id)
            if job is not None:
                job.update(fields)


def _to_agent_action(action: dict[str, Any]) -> Any:
    from executor.action import AgentAction

    return AgentAction.from_dict(action)


def _safe_path(path: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    candidate = candidate.resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError(f"路径必须在项目目录内: {path}")
    return candidate


manager = SessionManager()
