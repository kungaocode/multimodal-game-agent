"""级联感知路由：本地轻量模型优先，视觉大模型按需兜底（阶段 6/14 深化）。

用户方案落地：
    常规流程（打资源/捐兵）高度相似 → 先跑本地 YOLO 检测器 + ResNet 分类器
    （毫秒级、零成本、可离线）；
    只有置信度不足 / 关键对象缺失时，才调用 Qwen-VL 视觉大模型兜底；
    LLM 的成功样本自动回流到样例库（教师-学生蒸馏 + 主动学习），
    供后续微调本地模型，形成「本地为主 → 大模型兜底 → 回流再训练」闭环。

路由决策完全确定性、可解释（返回触发原因），不做「用 LLM 判断要不要调 LLM」。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from PIL import Image

from decision.resource_policy import RESOURCE_NAMES
from state.game_state import GameState

from .detector import Detector
from .game_state_extractor import GameStateExtractor
from .vision_model import VisionModel

# 打资源场景的兜底提示词：要求模型输出像素坐标，便于直接合并进 GameState
CASCADE_PROMPT = (
    "你是部落冲突游戏界面分析师。请识别图中的资源建筑（金矿、圣水收集器、储金罐、圣水瓶、"
    "暗黑重油罐、暗黑重油钻井）和重要 UI 元素（进攻按钮、返回按钮）。"
    '请用 JSON 输出：{"description": "一句话描述", "objects": [{"type": "建筑名", '
    '"coords": [x, y], "confidence": 0.0~1.0, "description": "细节"}]}。'
    "coords 必须是图片中的像素坐标（画面 1280x720），例如 [640, 360] 表示画面中心。"
)

# 路由模式
MODE_AUTO = "auto"   # 本地优先，规则触发才调 LLM
MODE_LOCAL = "local"  # 只用本地模型
MODE_LLM = "llm"     # 每次强制 LLM（用于验收 / 生成训练样本）

# 服务级默认模式（环境变量可覆盖）
CASCADE_MODE = os.getenv("CASCADE_MODE", MODE_AUTO)


@dataclass(frozen=True)
class CascadeConfig:
    """级联路由配置与触发阈值。"""

    mode: str = MODE_AUTO
    # 检测平均置信度低于该值 → 调 LLM
    min_avg_confidence: float = 0.5
    # 场景关键建筑：一个都没检出 → 调 LLM（打资源用 6 类资源建筑；捐兵场景可换成部落城堡）
    require_key_buildings: tuple[str, ...] = RESOURCE_NAMES
    # 样本回流目录（保存截图 + LLM 输出，供后续微调）
    samples_dir: str = "dataset/llm_samples"
    # 回流样本上限，超出后清理最旧
    max_samples: int = 500


@dataclass
class CascadeReport:
    """单次感知的路由报告（route: local / llm / local_no_llm）。"""

    route: str = "local"
    reasons: list[str] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    sample_saved: str | None = None  # 回流样本文件路径

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "reasons": self.reasons,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "sample_saved": self.sample_saved,
        }


class LlmSampleRecorder:
    """把 LLM 兜底样本（截图 + 提示词 + 模型输出）回流到磁盘，供微调/审计。

    目录结构：
        <samples_dir>/<timestamp>.jpg     原始截图（压缩到宽 1280）
        <samples_dir>/<timestamp>.json    提示词 + 模型输出 + 路由原因 + token 用量
    """

    def __init__(self, samples_dir: str | Path = "dataset/llm_samples", max_samples: int = 500) -> None:
        self.dir = Path(samples_dir)
        self.max_samples = max_samples

    def record(
        self,
        image: Image.Image,
        prompt: str,
        response: dict[str, Any] | None,
        reasons: Sequence[str],
        usage: dict[str, Any] | None = None,
        route: str = "llm",
    ) -> str:
        """保存一条样本，返回 meta 文件路径；删除最旧样本维持上限。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        meta_path = self.dir / f"{ts}.json"
        img_path = self.dir / f"{ts}.jpg"
        if img_path.exists():
            # 同秒重名：追加毫秒
            ts = f"{ts}_{time.time_ns() % 1_000_000:06d}"
            meta_path = self.dir / f"{ts}.json"
            img_path = self.dir / f"{ts}.jpg"

        img = image.convert("RGB")
        if img.width > 1280:
            img = img.resize((1280, int(1280 * img.height / img.width)), Image.LANCZOS)
        img.save(img_path, format="JPEG", quality=90)

        meta = {
            "timestamp": ts,
            "image": img_path.name,
            "route": route,
            "reasons": list(reasons),
            "prompt": prompt,
            "response": response,
            "usage": usage or {},
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._trim()
        return str(meta_path)

    def _trim(self) -> None:
        metas = sorted(self.dir.glob("*.json"))
        while len(metas) > self.max_samples:
            old = metas.pop(0)
            old.unlink(missing_ok=True)
            (self.dir / old.with_suffix(".jpg").name).unlink(missing_ok=True)

    def count(self) -> int:
        if not self.dir.exists():
            return 0
        return len(list(self.dir.glob("*.json")))


class CascadeStats:
    """级联路由累计统计（内存态），用于监控本地模型表现与 LLM 成本。"""

    def __init__(self) -> None:
        self.total = 0
        self.local = 0
        self.llm = 0
        self.reason_counts: dict[str, int] = {}
        self.llm_prompt_tokens = 0
        self.llm_completion_tokens = 0

    def record(self, report: CascadeReport) -> None:
        self.total += 1
        if report.route == "llm":
            self.llm += 1
            self.llm_prompt_tokens += report.prompt_tokens
            self.llm_completion_tokens += report.completion_tokens
            for reason in report.reasons:
                self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        else:
            self.local += 1

    def snapshot(self, recorder: LlmSampleRecorder | None = None) -> dict[str, Any]:
        llm_ratio = self.llm / self.total if self.total else 0.0
        return {
            "total": self.total,
            "local": self.local,
            "llm": self.llm,
            "llm_ratio": round(llm_ratio, 4),
            "llm_trigger_reasons": dict(
                sorted(self.reason_counts.items(), key=lambda kv: -kv[1])
            ),
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_completion_tokens": self.llm_completion_tokens,
            "samples_collected": recorder.count() if recorder else 0,
        }


def needs_llm(state: GameState, config: CascadeConfig) -> tuple[bool, list[str]]:
    """确定性路由决策：是否需要 LLM 兜底。返回 (是否需要, 触发原因列表)。

    - 一个建筑都没检出 → 触发
    - 平均置信度低于 min_avg_confidence → 触发
    - 场景关键建筑一个都没检出（如打资源时没有资源建筑）→ 触发
    """
    reasons: list[str] = []
    buildings = state.buildings
    if not buildings:
        reasons.append("本地模型未检出任何建筑")
        return True, reasons
    avg_conf = sum(b.confidence for b in buildings) / len(buildings)
    if avg_conf < config.min_avg_confidence:
        reasons.append(f"平均置信度 {avg_conf:.2f} < 阈值 {config.min_avg_confidence:.2f}")
    if not any(b.type in config.require_key_buildings for b in buildings):
        reasons.append("未检出场景关键建筑")
    return bool(reasons), reasons


def _llm_objects_to_buildings(state: GameState, objects: list[Any]) -> int:
    """把 LLM 输出的带像素坐标 objects 合并进 GameState.buildings（source 标注进 warnings 说明）。

    跳过 coords 缺失/非法（0,0）的项；返回合并数量。
    """
    merged = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        coords = obj.get("coords")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            continue
        try:
            cx, cy = int(coords[0]), int(coords[1])
        except (TypeError, ValueError):
            continue
        if cx <= 0 or cy <= 0 or not isinstance(obj.get("type"), str) or not obj["type"]:
            continue
        conf = float(obj.get("confidence") or 0.5)
        if not 0.0 <= conf <= 1.0:
            conf = 0.5
        state.add_building(
            building_type=obj["type"],
            position=(cx, cy),
            confidence=conf,
        )
        merged += 1
    return merged


class CascadeExtractor:
    """级联感知器：本地 GameStateExtractor + 按需 LLM 兜底 + 样本回流。"""

    def __init__(
        self,
        detector: Detector | None,
        classifier: Any | None,
        vision: VisionModel | None,
        config: CascadeConfig | None = None,
        recorder: LlmSampleRecorder | None = None,
        stats: CascadeStats | None = None,
    ) -> None:
        self.local = GameStateExtractor(detector=detector, classifier=classifier)
        self.vision = vision
        self.config = config or CascadeConfig()
        self.recorder = recorder or LlmSampleRecorder(self.config.samples_dir, self.config.max_samples)
        self.stats = stats or CascadeStats()

    def extract(
        self,
        image: Image.Image | str | Path,
        image_path: str | None = None,
        force_llm: bool = False,
        mode: str | None = None,
    ) -> tuple[GameState, CascadeReport]:
        """执行一次级联感知：本地优先，按需 LLM 兜底。返回 (GameState, 路由报告)。

        mode 覆盖本次请求的路由模式（auto/local/llm），不影响实例配置。
        """
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            pil = Image.fromarray(image).convert("RGB")

        state = self.local.extract(pil, image_path=image_path)
        report = CascadeReport()

        mode = mode or self.config.mode
        need, reasons = needs_llm(state, self.config)

        if mode == MODE_LOCAL:
            report.route = "local"
        elif mode == MODE_LLM:
            need, reasons = True, ["CASCADE_MODE=llm 强制调用"]
            report.route = "llm"
        else:  # MODE_AUTO
            report.reasons = reasons
            report.route = "llm" if need else "local"

        if report.route == "llm" and self.vision is not None:
            try:
                parsed = self.vision.chat(pil, prompt=CASCADE_PROMPT, include_usage=True)
                usage = parsed.pop("_usage", {}) or {} if isinstance(parsed, dict) else {}
                report.llm_calls = 1
                report.prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                report.completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                objects = parsed.get("objects", []) if isinstance(parsed, dict) else []
                merged = _llm_objects_to_buildings(state, objects)
                # 保留原始 LLM 输出便于审计
                state.screen_text.append(
                    {
                        "source": "vision_model(cascade)",
                        "description": parsed.get("description", ""),
                        "objects": objects,
                    }
                )
                if merged:
                    state.warnings.append(
                        f"cascade: LLM 兜底补入 {merged} 个建筑（坐标为模型估算）"
                    )
                report.sample_saved = self.recorder.record(
                    pil,
                    CASCADE_PROMPT,
                    parsed,
                    reasons,
                    usage=usage,
                    route="llm",
                )
            except Exception as exc:  # noqa: BLE001 - 兜底失败不中断，保持本地结果
                state.warnings.append(f"cascade: LLM 兜底失败（{exc}），使用本地结果")
                report.route = "local"
        elif report.route == "llm":
            state.warnings.append(
                "cascade: 本地模型置信度不足但未配置 VISION_MODEL_*，使用本地结果"
            )
            report.route = "local_no_llm"

        self.stats.record(report)
        return state, report


def build_default(config: CascadeConfig | None = None) -> CascadeExtractor:
    """便捷工厂：用默认权重路径构造级联感知器（YOLO + ResNet + 可选 vision）。"""
    from pathlib import Path

    detector = Detector("runs/detect/fsm_v4/weights/best.pt")
    from .classifier import BuildingClassifier

    classifier = BuildingClassifier.load("runs/classifier/best.pt")
    vision = None
    probe = VisionModel()
    if probe.base_url:
        vision = probe
    return CascadeExtractor(detector, classifier, vision, config=config)
