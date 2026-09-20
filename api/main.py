"""感知 HTTP 服务：截图 -> 结构化游戏状态（GameState）。

重型模型（YOLO 检测器 / PaddleOCR / 视觉大模型）全部懒加载，服务启动即快；
未训练/未配置的组件会优雅降级，并在响应的 warnings 中说明。

组件开关（环境变量）：
    DETECTOR_MODEL    训练好的 YOLO 权重路径；为空或文件不存在则禁用检测
    CLASSIFIER_MODEL  训练好的 ResNet18 分类器；对每个检测框补等级（level）
    ENABLE_OCR        设为 1 启用 PaddleOCR（首次使用会下载模型）
    VISION_MODEL_BASE_URL / VISION_MODEL_API_KEY / VISION_MODEL_NAME
                     配置后启用 Qwen-VL 类视觉大模型（模型名默认 qwen-vl-max）
"""

from __future__ import annotations

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from api.planner import build_plan, candidates_from_llm_steps, llm_plan_prompt
from perception import BuildingClassifier, Detector, GameStateExtractor, OCRReader, VisionModel
from perception.cascade import (
    CASCADE_MODE,
    CascadeConfig,
    CascadeExtractor,
)
from state.game_state import GameState
from api.sessions import RetrainRequest, SessionStartRequest, manager

DETECTOR_MODEL = Path(os.getenv("DETECTOR_MODEL", "runs/detect/fsm_v4/weights/best.pt"))
CLASSIFIER_MODEL = Path(os.getenv("CLASSIFIER_MODEL", "runs/classifier/best.pt"))
ENABLE_OCR = os.getenv("ENABLE_OCR", "0") == "1"


class _Runtime:
    """懒加载的感知组件集合。"""

    def __init__(self) -> None:
        self.detector: Detector | None = None
        self.classifier: BuildingClassifier | None = None
        self.ocr: OCRReader | None = None
        self.vision: VisionModel | None = None
        self.cascade: CascadeExtractor | None = None

    def _load_detector(self) -> Detector | None:
        if self.detector is None and DETECTOR_MODEL.exists():
            self.detector = Detector(DETECTOR_MODEL)
        return self.detector

    def _load_classifier(self) -> BuildingClassifier | None:
        if self.classifier is None and CLASSIFIER_MODEL.exists():
            self.classifier = BuildingClassifier.load(CLASSIFIER_MODEL)
        return self.classifier

    def _load_ocr(self) -> OCRReader | None:
        if self.ocr is None and ENABLE_OCR:
            self.ocr = OCRReader(lang="ch", use_gpu=False)
        return self.ocr

    def _load_vision(self) -> VisionModel | None:
        if self.vision is None:
            model = VisionModel()
            if model.base_url:
                self.vision = model
        return self.vision

    def extractor(self) -> GameStateExtractor:
        return GameStateExtractor(
            detector=self._load_detector(),
            classifier=self._load_classifier(),
            ocr=self._load_ocr(),
            vision_model=self._load_vision(),
        )

    def get_cascade(self) -> CascadeExtractor:
        """级联感知器（本地优先 + 按需 LLM 兜底），跨请求复用统计与样本回流。

        注意：方法名不能与字段名（self.cascade）同名，否则实例属性会遮蔽方法。
        """
        if self.cascade is None:
            self.cascade = CascadeExtractor(
                detector=self._load_detector(),
                classifier=self._load_classifier(),
                vision=self._load_vision(),
                config=CascadeConfig(mode=CASCADE_MODE),
            )
        return self.cascade

    def status(self) -> dict:
        vision = self._load_vision()
        return {
            "detector": {
                "model_path": str(DETECTOR_MODEL),
                "enabled": DETECTOR_MODEL.exists(),
                "loaded": self.detector is not None,
            },
            "classifier": {
                "model_path": str(CLASSIFIER_MODEL),
                "enabled": CLASSIFIER_MODEL.exists(),
                "loaded": self.classifier is not None,
            },
            "ocr": {"enabled": ENABLE_OCR, "loaded": self.ocr is not None},
            "vision_model": {
                "configured": vision is not None,
                "loaded": self.vision is not None,
            },
        }


runtime = _Runtime()
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # 关闭重型组件释放资源（无 close 的组件自动跳过）
    for component in (runtime.detector, runtime.ocr):
        close = getattr(component, "close", None)
        if callable(close):
            close()


app = FastAPI(title="Clash CoC Perception API", version="0.1.0", lifespan=lifespan)


class PerceiveBase64Request(BaseModel):
    image_base64: str = Field(description="Base64 编码的图片（jpg/png）")
    filename: str | None = None


class PlanRequest(BaseModel):
    game_state: GameState = Field(description="结构化游戏状态（/perceive 的输出）")
    full_resources: list[str] = Field(
        default_factory=list,
        description='己方已满的资源类型，如 ["gold", "elixir"]；对应资源建筑会被跳过',
    )
    top_k: int | None = Field(default=None, ge=1, description="只返回得分最高的前 N 个候选")
    image_size: list[int] | None = Field(
        default=None, description="画面尺寸 [宽, 高]，默认 [1280, 720]"
    )
    image_base64: str | None = Field(
        default=None,
        description="可选：原始截图 base64。提供且已配置 VISION_MODEL_* 时，附加 Qwen-VL 生成的 LLM 计划",
    )


def _load_image(data: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}") from exc


def _run_pipeline(image: Image.Image, filename: str | None) -> GameState:
    return runtime.extractor().extract(image, image_path=filename)


def _run_cascade(image: Image.Image, filename: str | None, mode: str | None) -> GameState:
    """级联感知：返回 GameState，并把路由信息写入 warnings（不改变响应 schema）。"""
    state, report = runtime.get_cascade().extract(image, image_path=filename, mode=mode)
    if report.route == "llm":
        state.warnings.append(f"cascade: 使用视觉大模型兜底（{len(report.reasons)} 条原因）")
    elif report.route == "local_no_llm":
        state.warnings.append("cascade: 需要 LLM 兜底但未配置，使用本地结果")
    return state


def _normalize_mode(mode: str | None) -> str | None:
    if mode in (CASCADE_MODE, "auto"):
        return None  # 用服务端默认
    if mode in ("local", "llm"):
        return mode
    raise HTTPException(status_code=400, detail=f"mode 只能是 auto/local/llm，收到: {mode!r}")


@app.get("/")
def root() -> dict:
    return {
        "service": "Clash CoC Perception API",
        "endpoints": [
            "GET /health",
            "POST /perceive?mode=auto|local|llm",
            "POST /perceive/base64",
            "POST /plan",
            "GET /cascade/stats",
            "GET /static/index.html",
            "GET /session/capabilities",
            "GET /session/videos?module=farm|donation",
            "POST /session/upload/video?module=farm|donation",
            "POST /session/start",
            "GET /session/{session_id}",
            "POST /session/{session_id}/stop",
            "POST /session/retrain",
            "GET /session/retrain/{job_id}",
        ],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "components": runtime.status()}


@app.post("/perceive", response_model=GameState)
def perceive(file: UploadFile = File(...), mode: str = Query("auto")) -> GameState:
    """上传截图（multipart form，字段名 file），返回结构化游戏状态。

    mode: auto（默认，级联：本地优先+LLM 兜底）| local（只用本地模型）| llm（强制大模型）
    """
    data = file.file.read()
    image = _load_image(data)
    return _run_cascade(image, file.filename, _normalize_mode(mode))


@app.post("/perceive/base64", response_model=GameState)
def perceive_base64(req: PerceiveBase64Request, mode: str = Query("auto")) -> GameState:
    """以 JSON 上传 base64 图片（Android/Harmony 客户端常用），返回结构化游戏状态。

    mode: auto|local|llm，语义同 /perceive。
    """
    try:
        data = base64.b64decode(req.image_base64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"base64 解码失败: {exc}") from exc
    image = _load_image(data)
    return _run_cascade(image, req.filename, _normalize_mode(mode))


@app.post("/plan")
def plan(req: PlanRequest) -> dict:
    """接收 GameState → 规则层候选排序（ResourcePolicy）+ 可选 LLM 作战计划。"""
    image_size = tuple(req.image_size) if req.image_size else None
    result = build_plan(
        req.game_state,
        full_resources=req.full_resources,
        image_size=image_size or (1280, 720),
        top_k=req.top_k,
    )

    llm_plan: dict | None = None
    if req.image_base64:
        vision = runtime._load_vision()
        if vision is None:
            result["warnings"].append("未配置 VISION_MODEL_BASE_URL/API_KEY，跳过 LLM 计划（规则层仍可用）")
        else:
            try:
                data = base64.b64decode(req.image_base64)
                image = _load_image(data)
                prompt = llm_plan_prompt(result["policy"])
                parsed = vision.chat(image, prompt=prompt)
                steps = parsed.get("steps", []) if isinstance(parsed, dict) else []
                llm_plan = {
                    "summary": (parsed.get("summary") if isinstance(parsed, dict) else "") or "",
                    "steps": steps,
                }
                # 规则层（YOLO）没检出资源建筑时，用 LLM 步骤补充候选（坐标是估算值）
                if not result["policy"]["candidates"]:
                    llm_candidates = candidates_from_llm_steps(steps)
                    if llm_candidates:
                        result["policy"]["candidates"] = llm_candidates
                        result["warnings"].append(
                            "规则层未检出资源建筑，候选由视觉大模型提供（坐标为估算值）"
                        )
            except Exception as exc:  # noqa: BLE001
                result["warnings"].append(f"vision_model: {exc}")

    return {"policy": result["policy"], "llm_plan": llm_plan, "warnings": result["warnings"]}


@app.get("/cascade/stats")
def cascade_stats() -> dict:
    """级联路由累计统计：本地/LLM 次数、LLM 触发率、触发原因、token 用量、回流样本数。"""
    ext = runtime.get_cascade()
    return ext.stats.snapshot(ext.recorder)


@app.get("/session/capabilities")
def session_capabilities() -> dict:
    """移动端启动前检查模型、教练 API 和 ADB 可用性。"""
    return manager.capabilities()


@app.get("/session/videos")
def session_videos(module: str = Query("farm")) -> dict:
    if module not in {"farm", "donation"}:
        raise HTTPException(status_code=400, detail="module 只能是 farm 或 donation")
    return {"module": module, "videos": manager.list_videos(module)}


@app.post("/session/upload/video")
def upload_video(module: str = Query("farm"), file: UploadFile = File(...)) -> dict:
    """从手机上传录屏，保存到对应模块的视频库后可做 coach dry-run。"""
    try:
        data = file.file.read()
        path = manager.save_video(module, file.filename or "upload.mp4", data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"module": module, "path": path}


@app.post("/session/start")
def session_start(req: SessionStartRequest) -> dict:
    """启动一个 coach-guided 会话；请求立刻返回，进度用 /session/{id} 轮询。"""
    try:
        return manager.start(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/{session_id}")
def session_status(session_id: str, steps_limit: int = Query(30, ge=1, le=100)) -> dict:
    try:
        return manager.get(session_id, steps_limit=steps_limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc


@app.post("/session/{session_id}/stop")
def session_stop(session_id: str) -> dict:
    try:
        return manager.request_stop(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="会话不存在") from exc


@app.post("/session/retrain")
def session_retrain(req: RetrainRequest) -> dict:
    """从教练会话提取回流数据，并在后台重训 YOLO + 状态分类器。"""
    try:
        return manager.start_retrain(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/session/retrain/{job_id}")
def retrain_status(job_id: str) -> dict:
    try:
        return manager.retrain_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="训练任务不存在") from exc


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
