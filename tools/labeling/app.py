"""半自动标注工具：检测器预标注 + 人工修正，产出 YOLO 格式标注。

用法：
    python -m tools.labeling.app --port 8001
浏览器打开 http://127.0.0.1:8001

工作流：
    1. 打开一张真实截图，检测器自动画候选框（类型已猜）。
    2. 人工修正：删误检、改标签、拖拽/缩放框、手动加框。
    3. 点保存，写入 YOLO 格式标注（class_id cx cy w h，归一化）。
    4. 用 dataset/real_labels/ 微调检测器（见 finetune.py）。

配置（命令行参数）：
    --raw   真实截图目录，默认 dataset/game_picture
    --out   标注输出目录，默认 dataset/real_labels
    --labels 类别映射（LabelMap JSON），默认 dataset/detection/labels.json
    --model 检测器权重，默认 runs/detect/train/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from perception.dataset import LabelMap
from perception.detector import Detector

# 模块级默认值，可在命令行覆盖
RAW_DIR = Path("dataset/game_picture")
OUT_DIR = Path("dataset/real_labels")
LABELS_PATH = Path("dataset/farm_labels.json")
MODEL_PATH = Path("runs/detect/train/weights/best.pt")

DETECTOR_CONF = 0.35


class Box(BaseModel):
    class_id: int
    x1: int
    y1: int
    x2: int
    y2: int


class Annotation(BaseModel):
    boxes: list[Box]


class _AppState:
    def __init__(self) -> None:
        self.detector: Detector | None = None


state = _AppState()
label_map: LabelMap | None = None


def _load_label_map() -> LabelMap:
    global label_map
    if label_map is None:
        label_map = LabelMap.load(LABELS_PATH)
    return label_map


def _load_detector() -> Detector | None:
    if state.detector is None and MODEL_PATH.exists():
        state.detector = Detector(MODEL_PATH)
    return state.detector


def _resolve_image(name: str) -> Path:
    """把文件名解析到 raw 目录内，防止路径穿越。"""
    candidate = (RAW_DIR / name).resolve()
    raw_root = RAW_DIR.resolve()
    if not candidate.is_file() or raw_root not in candidate.parents:
        raise HTTPException(status_code=404, detail="image not found")
    return candidate


def _list_images() -> list[str]:
    if not RAW_DIR.is_dir():
        return []
    names: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        names.extend(sorted(p.name for p in RAW_DIR.glob(ext)))
    return names


def _saved_boxes(name: str) -> list[Box] | None:
    """读已保存的 YOLO 标注；没有则返回 None。"""
    label_path = OUT_DIR / "labels" / f"{Path(name).stem}.txt"
    if not label_path.exists():
        return None
    width, height = _image_size(name)
    boxes: list[Box] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, xc, yc, w, h = map(float, parts)
        # 用 round 而非 int：浮点误差（如 19.999…）会被 int 截断成 19
        x1 = round((xc - w / 2) * width)
        y1 = round((yc - h / 2) * height)
        x2 = round((xc + w / 2) * width)
        y2 = round((yc + h / 2) * height)
        boxes.append(Box(class_id=int(class_id), x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


def _image_size(name: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(_resolve_image(name)) as img:
        return img.size


def _detect_boxes(name: str) -> list[Box]:
    """用检测器预标注；模型不可用或失败返回空列表。"""
    detector = _load_detector()
    if detector is None:
        return []
    try:
        from PIL import Image

        width, height = _image_size(name)
        results = detector.predict(_resolve_image(name), conf=DETECTOR_CONF)
        lm = _load_label_map()
        boxes: list[Box] = []
        for r in results:
            try:
                class_id = lm.id_for(r.class_name)
            except KeyError:
                continue
            boxes.append(Box(class_id=class_id, x1=r.bbox[0], y1=r.bbox[1], x2=r.bbox[2], y2=r.bbox[3]))
        return boxes
    except Exception:  # noqa: BLE001 - 标注工具容错：预标注失败就空框手工画
        return []


def get_boxes(name: str) -> dict:
    """获取某张截图的标注：已有保存则加载，否则用检测器预标注。"""
    _resolve_image(name)  # 校验存在
    saved = _saved_boxes(name)
    box_list = saved if saved is not None else _detect_boxes(name)
    width, height = _image_size(name)
    lm = _load_label_map()
    return {
        "name": name,
        "width": width,
        "height": height,
        "saved": saved is not None,
        "boxes": [
            {"class_id": b.class_id, "name": lm.names.get(b.class_id, "?"), "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2}
            for b in box_list
        ],
    }


def save_annotation(name: str, ann: Annotation) -> dict:
    """把人工修正后的框写入 YOLO 归一化标注。"""
    _resolve_image(name)
    width, height = _image_size(name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "labels").mkdir(parents=True, exist_ok=True)
    label_path = OUT_DIR / "labels" / f"{Path(name).stem}.txt"
    lines: list[str] = []
    for b in ann.boxes:
        x1, y1, x2, y2 = b.x1, b.y1, b.x2, b.y2
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        lines.append(f"{b.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    _load_label_map().save(OUT_DIR / "labels.json")
    return {"saved": len(ann.boxes), "path": str(label_path)}


def build_app() -> FastAPI:
    app = FastAPI(title="CoC 半自动标注工具", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        static = Path(__file__).parent / "static" / "index.html"
        return static.read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta() -> dict:
        lm = _load_label_map()
        return {
            "classes": {str(cid): name for cid, name in lm.names.items()},
            "images": _list_images(),
        }

    @app.get("/api/boxes/{name}")
    def boxes(name: str) -> dict:
        return get_boxes(name)

    @app.get("/img/{name}")
    def image(name: str) -> FileResponse:
        return FileResponse(_resolve_image(name))

    @app.post("/api/save/{name}")
    def save(name: str, ann: Annotation) -> dict:
        return save_annotation(name, ann)

    return app


def main(argv: list[str] | None = None) -> int:
    global RAW_DIR, OUT_DIR, LABELS_PATH, MODEL_PATH
    parser = argparse.ArgumentParser(description="CoC 半自动标注工具")
    parser.add_argument("--raw", default=str(RAW_DIR), type=Path)
    parser.add_argument("--out", default=str(OUT_DIR), type=Path)
    parser.add_argument("--labels", default=str(LABELS_PATH), type=Path)
    parser.add_argument("--model", default=str(MODEL_PATH), type=Path)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args(argv)
    RAW_DIR, OUT_DIR, LABELS_PATH, MODEL_PATH = args.raw, args.out, args.labels, args.model

    import uvicorn

    print(f"标注工具: http://127.0.0.1:{args.port}")
    print(f"  原始截图: {RAW_DIR}  标注输出: {OUT_DIR}")
    print(f"  检测器: {MODEL_PATH}  类别: {LABELS_PATH}")
    uvicorn.run(build_app(), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
