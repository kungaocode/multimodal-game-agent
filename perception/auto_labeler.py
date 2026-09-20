"""Shared auto-labeling: OCR + LLM -> YOLO format annotation boxes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from perception.fsm_labels import (
    EXTENDED_FSM_ORDER,
    EXTENDED_NAME_TO_ID,
    build_llm_label_prompt,
    extended_class_id_for_button_text,
)

logger = logging.getLogger(__name__)


@dataclass
class YoloBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    confidence: float = 1.0
    source: str = "manual"

    def to_line(self) -> str:
        # Keep every YOLO coordinate inside the image.  LLM boxes can extend
        # past the right/bottom edge, which Ultralytics rejects as corrupt.
        w = min(max(self.w, 1e-6), 1.0)
        h = min(max(self.h, 1e-6), 1.0)
        cx = min(max(self.cx, w / 2), 1.0 - w / 2)
        cy = min(max(self.cy, h / 2), 1.0 - h / 2)
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class FrameLabel:
    image_path: str
    boxes: list[YoloBox] = field(default_factory=list)
    state: str | None = None
    ocr_boxes: int = 0
    llm_boxes: int = 0
    sources: list[str] = field(default_factory=list)

    @property
    def total_boxes(self) -> int:
        return len(self.boxes)


def ocr_label_frame(
    image: Image.Image | Path | str,
    ocr: Any,
    pad: int = 12,
    min_conf: float = 0.5,
    use_extended: bool = True,
) -> list[YoloBox]:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    w, h = img.size
    results = ocr.read_text(img)
    boxes: list[YoloBox] = []
    for r in results:
        if r.confidence < min_conf:
            continue
        if use_extended:
            cid = extended_class_id_for_button_text(r.text)
        else:
            from perception.fsm_labels import class_id_for_button_text
            cid = class_id_for_button_text(r.text)
        if cid is None:
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(YoloBox(
            class_id=cid,
            cx=(x1 + x2) / 2 / w,
            cy=(y1 + y2) / 2 / h,
            w=(x2 - x1) / w,
            h=(y2 - y1) / h,
            confidence=r.confidence,
            source="ocr",
        ))
    return boxes


def _parse_llm_bbox(
    obj: dict[str, Any], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    bbox = obj.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            x1 = min(max(x1, 0.0), img_w)
            y1 = min(max(y1, 0.0), img_h)
            x2 = min(max(x2, 0.0), img_w)
            y2 = min(max(y2, 0.0), img_h)
            if x2 > x1 and y2 > y1 and 0 <= x1 < img_w and 0 <= y1 < img_h:
                cx = (x1 + x2) / 2 / img_w
                cy = (y1 + y2) / 2 / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                return cx, cy, min(bw, 1.0), min(bh, 1.0)
        except (TypeError, ValueError):
            pass
    coords = obj.get("coords") or obj.get("position_estimate")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            cx_px, cy_px = float(coords[0]), float(coords[1])
            if 0 <= cx_px < img_w and 0 <= cy_px < img_h:
                obj_type = obj.get("type", "")
                from perception.fsm_labels import EXTENDED_PRIOR_BOX
                pw, ph = EXTENDED_PRIOR_BOX.get(obj_type, (0.10, 0.10))
                return cx_px / img_w, cy_px / img_h, pw, ph
        except (TypeError, ValueError):
            pass
    return None


def llm_label_frame(
    image: Image.Image | Path | str,
    vision: Any,
    use_extended: bool = True,
) -> tuple[list[YoloBox], str | None]:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.convert("RGB")
    w, h = img.size
    prompt = build_llm_label_prompt(w, h)
    try:
        parsed = vision.chat(img, prompt=prompt)
    except Exception as exc:
        logger.warning("LLM label failed: %s", exc)
        return [], None
    if not isinstance(parsed, dict):
        return [], None
    state = parsed.get("state")
    objects = parsed.get("objects", [])
    boxes: list[YoloBox] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("type", "")
        if not obj_type:
            continue
        if use_extended:
            cid = EXTENDED_NAME_TO_ID.get(obj_type)
        else:
            from perception.fsm_labels import FSM_NAME_TO_ID
            cid = FSM_NAME_TO_ID.get(obj_type)
        if cid is None:
            continue
        parsed_bbox = _parse_llm_bbox(obj, w, h)
        if parsed_bbox is None:
            continue
        cx, cy, bw, bh = parsed_bbox
        conf = float(obj.get("confidence") or 0.5)
        if not 0.0 <= conf <= 1.0:
            conf = 0.5
        boxes.append(YoloBox(
            class_id=cid, cx=cx, cy=cy, w=bw, h=bh,
            confidence=conf, source="llm",
        ))
    return boxes, state


def merge_labels(
    ocr_boxes: list[YoloBox],
    llm_boxes: list[YoloBox],
    iou_threshold: float = 0.5,
) -> list[YoloBox]:
    merged = list(ocr_boxes)
    for lb in llm_boxes:
        is_dup = False
        for ob in merged:
            if ob.class_id != lb.class_id:
                continue
            if _box_iou(ob, lb) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            merged.append(lb)
    return merged


def _box_iou(a: YoloBox, b: YoloBox) -> float:
    ax1, ay1 = a.cx - a.w / 2, a.cy - a.h / 2
    ax2, ay2 = a.cx + a.w / 2, a.cy + a.h / 2
    bx1, by1 = b.cx - b.w / 2, b.cy - b.h / 2
    bx2, by2 = b.cx + b.w / 2, b.cy + b.h / 2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_frame_difference(
    prev: np.ndarray, curr: np.ndarray, resize: tuple[int, int] = (64, 36)
) -> float:
    p = np.asarray(Image.fromarray(prev).resize(resize), dtype=np.float32)
    c = np.asarray(Image.fromarray(curr).resize(resize), dtype=np.float32)
    return float(np.mean(np.abs(p - c)))


def extract_frames(
    video_path: Path | str,
    fps: float = 1.0,
    diff_threshold: float = 8.0,
    max_frames: int = 2000,
) -> list[Image.Image]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(video_fps / fps))
    frames: list[Image.Image] = []
    prev_small: np.ndarray | None = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (64, 36))
            if prev_small is not None:
                diff = float(np.mean(np.abs(
                    small.astype(np.float32) - prev_small.astype(np.float32)
                )))
                if diff < diff_threshold:
                    frame_idx += 1
                    continue
            prev_small = small
            pil = Image.fromarray(rgb)
            frames.append(pil)
            if len(frames) >= max_frames:
                logger.warning("Max frames reached: %d", max_frames)
                break
        frame_idx += 1
    cap.release()
    logger.info("Extracted %d deduplicated frames from %s", len(frames), video_path)
    return frames


def auto_label_frame(
    image: Image.Image | Path | str,
    ocr: Any | None = None,
    vision: Any | None = None,
    use_extended: bool = True,
    ocr_pad: int = 12,
) -> FrameLabel:
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        path_str = str(image)
    else:
        img = image
        path_str = "<inline>"
    label = FrameLabel(image_path=path_str)
    ocr_boxes: list[YoloBox] = []
    llm_boxes: list[YoloBox] = []
    if ocr is not None:
        ocr_boxes = ocr_label_frame(img, ocr, pad=ocr_pad, use_extended=use_extended)
        label.ocr_boxes = len(ocr_boxes)
        label.sources.append("ocr")
    if vision is not None:
        llm_boxes, state = llm_label_frame(img, vision, use_extended=use_extended)
        label.llm_boxes = len(llm_boxes)
        label.state = state
        label.sources.append("llm")
    label.boxes = merge_labels(ocr_boxes, llm_boxes)
    return label


def save_yolo_dataset(
    labels: list[FrameLabel],
    images: list[Image.Image],
    out_dir: Path | str,
    val_ratio: float = 0.2,
    class_names: list[str] | None = None,
    seed: int = 42,
) -> Path:
    import random

    from perception.dataset import LabelMap, build_yolo_dataset_yaml

    if class_names is None:
        class_names = EXTENDED_FSM_ORDER
    out = Path(out_dir)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    indices = list(range(len(labels)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(len(indices) * val_ratio)) if indices else 0
    state_labels: list[dict[str, Any]] = []
    for i, idx in enumerate(indices):
        lab = labels[idx]
        img = images[idx]
        split = "val" if i < n_val else "train"
        img_name = f"frame_{idx:06d}.jpg"
        img.save(out / "images" / split / img_name, format="JPEG", quality=92)
        lab_path = out / "labels" / split / f"frame_{idx:06d}.txt"
        lines = [b.to_line() for b in lab.boxes]
        lab_path.write_text("\n".join(lines), encoding="utf-8")
        state_labels.append({
            "frame": f"images/{split}/{img_name}",
            "state": lab.state,
            "detections": [
                {"class_id": b.class_id, "confidence": b.confidence}
                for b in lab.boxes
            ],
        })
    label_map = LabelMap.from_list(class_names)
    label_map.save(out / "labels.json")
    yaml_path = build_yolo_dataset_yaml(
        root=out,
        train_dir="images/train",
        val_dir="images/val",
        label_map=label_map,
        output_path=out / "dataset.yaml",
    )
    logger.info("Saved YOLO dataset to %s (%d train / %d val)", out, len(indices) - n_val, n_val)
    (out / "state_labels.json").write_text(
        json.dumps(state_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out
