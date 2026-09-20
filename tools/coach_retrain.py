"""Retrain local models from coach session data.

Reads coach_sessions/<timestamp>/ directories, extracts:
  1. YOLO training data: screenshots + coach-corrected bounding boxes (missed_objects)
     merged with local detections (already have boxes)
  2. State labels for StateClassifier: coach's corrected_state per frame

Usage:
    # Retrain from all sessions in a directory:
    python -m tools.coach_retrain --sessions dataset/coach_sessions --out dataset/coach_training

    # Then train YOLO:
    python -m tools.labeling.finetune_fsm --buttons dataset/coach_training/yolo --base runs/detect/fsm/weights/best.pt

    # Then train state classifier:
    python -m decision.state_classifier --train --data dataset/coach_training/state_labels.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

from PIL import Image

from perception.dataset import LabelMap, build_yolo_dataset_yaml
from perception.fsm_labels import EXTENDED_FSM_ORDER, EXTENDED_NAME_TO_ID, EXTENDED_PRIOR_BOX

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COACH_TYPE_ALIASES = {
    "进攻！按钮": "进攻按钮",
    "确认增援按钮（绿色）": "捐赠确认按钮",
    "增援按钮（主请求区）": "增援按钮",
    "请求增援消息": "请求条目",
    "增援请求消息": "请求条目",
    "部队选择栏": "兵种选择栏",
    "兵种选择栏": "兵种选择栏",
}


def _clamp_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    w, h = image_size
    x1 = max(0.0, min(float(x1), float(x2)))
    x2 = min(float(w), max(x1, x2))
    y1 = max(0.0, min(float(y1), float(y2)))
    y2 = min(float(h), max(y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _coach_bbox_to_image(
    obj: dict[str, object],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Convert coach output to actual pixels, supporting legacy and normalized values."""
    w, h = image_size
    bbox = obj.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            values = [float(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            values = []
        if len(values) == 4 and all(math.isfinite(v) for v in values):
            if all(0.0 <= v <= 1.0 for v in values):
                values = [values[0] * w, values[1] * h, values[2] * w, values[3] * h]
            else:
                # Coach sessions before the normalized-protocol fix used a fixed
                # 1280x960 logical canvas even for cropped screenshots.
                values = [
                    values[0] * w / 1280.0,
                    values[1] * h / 960.0,
                    values[2] * w / 1280.0,
                    values[3] * h / 960.0,
                ]
            return _clamp_bbox((values[0], values[1], values[2], values[3]), image_size)

    coords = obj.get("coords")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        try:
            cx, cy = float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            return None
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0:
            cx, cy = cx * w, cy * h
        else:
            cx, cy = cx * w / 1280.0, cy * h / 960.0
        pw, ph = EXTENDED_PRIOR_BOX.get(str(obj.get("type", "")), (0.10, 0.10))
        return _clamp_bbox((cx - pw * w / 2, cy - ph * h / 2, cx + pw * w / 2, cy + ph * h / 2), image_size)
    return None


def _local_bbox_to_image(
    bbox: object,
    image_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    return _clamp_bbox((x1, y1, x2, y2), image_size)


def _find_sessions(sessions_dir: Path) -> list[Path]:
    """Find all session directories (must contain session_summary.json)."""
    if not sessions_dir.exists():
        return []
    return sorted(
        d for d in sessions_dir.iterdir()
        if d.is_dir() and (d / "session_summary.json").exists()
    )


def extract_yolo_data(
    session_dirs: list[Path],
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> int:
    """Extract YOLO training data from coach sessions.

    Each step produces:
      - image: screenshot
      - label: local detections (from *_local.json) + coach missed_objects (from *_coach.json)
    """
    import random

    out_img_train = out_dir / "yolo" / "images" / "train"
    out_lab_train = out_dir / "yolo" / "labels" / "train"
    out_img_val = out_dir / "yolo" / "images" / "val"
    out_lab_val = out_dir / "yolo" / "labels" / "val"
    for d in (out_img_train, out_lab_train, out_img_val, out_lab_val):
        d.mkdir(parents=True, exist_ok=True)

    all_samples: list[tuple[Path, Path, list[str]]] = []  # (img_path, label_lines, source)

    for session_dir in session_dirs:
        for local_json in sorted(session_dir.glob("step_*_local.json")):
            prefix = local_json.stem.replace("_local", "")
            img_path = session_dir / f"{prefix}_screenshot.jpg"
            if not img_path.exists():
                continue
            coach_json = session_dir / f"{prefix}_coach.json"

            # Parse local detections
            local_data = json.loads(local_json.read_text(encoding="utf-8"))
            detections = local_data.get("detections", [])
            img = Image.open(img_path)
            w, h = img.size

            label_lines: list[str] = []
            for det in detections:
                obj_type = det.get("type", "")
                cid = EXTENDED_NAME_TO_ID.get(obj_type)
                if cid is None:
                    continue
                bbox = _local_bbox_to_image(det.get("bbox"), (w, h))
                if bbox is None:
                    continue
                cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
                bw, bh = (bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h
                label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # Add coach missed_objects (these are the valuable corrections)
            if coach_json.exists():
                coach_data = json.loads(coach_json.read_text(encoding="utf-8"))
                for obj in coach_data.get("missed_objects", []):
                    if not isinstance(obj, dict):
                        continue
                    obj_type = obj.get("type", "")
                    obj_type = COACH_TYPE_ALIASES.get(str(obj_type), str(obj_type))
                    cid = EXTENDED_NAME_TO_ID.get(obj_type)
                    if cid is None:
                        continue
                    bbox = _coach_bbox_to_image(obj, (w, h))
                    if bbox is None:
                        continue
                    cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
                    bw, bh = (bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h
                    label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if label_lines:
                unique_stem = f"{session_dir.name}_{prefix}"
                all_samples.append((img_path, out_dir / "_tmp" / f"{unique_stem}.txt", label_lines))

    # Shuffle and split
    random.Random(seed).shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio)) if all_samples else 0
    count = 0
    for i, (img_path, label_tmp, label_lines) in enumerate(all_samples):
        split = "val" if i < n_val else "train"
        img_dst = (out_img_val if split == "val" else out_img_train) / f"{label_tmp.stem}{img_path.suffix}"
        lab_dst = (out_lab_val if split == "val" else out_lab_train) / f"{label_tmp.stem}.txt"
        if not img_dst.exists():
            shutil.copy2(img_path, img_dst)
        lab_dst.write_text("\n".join(label_lines), encoding="utf-8")
        count += 1

    # Save labels.json
    label_map = LabelMap.from_list(EXTENDED_FSM_ORDER)
    label_map.save(out_dir / "yolo" / "labels.json")
    build_yolo_dataset_yaml(
        root=(out_dir / "yolo").resolve(),
        train_dir=out_img_train.resolve(),
        val_dir=out_img_val.resolve(),
        label_map=label_map,
        output_path=out_dir / "yolo" / "dataset.yaml",
    )
    logger.info("Extracted %d YOLO samples (%d train / %d val)", count, count - n_val, n_val)
    return count


def extract_state_labels(
    session_dirs: list[Path],
    out_dir: Path,
) -> int:
    """Extract state labels for StateClassifier training.

    Uses coach's corrected_state (higher quality) or local state as fallback.
    Output: state_labels.json
    """
    samples: list[dict] = []
    for session_dir in session_dirs:
        for local_json in sorted(session_dir.glob("step_*_local.json")):
            prefix = local_json.stem.replace("_local", "")
            coach_json = session_dir / f"{prefix}_coach.json"

            local_data = json.loads(local_json.read_text(encoding="utf-8"))
            detections = local_data.get("detections", [])

            # Prefer coach's corrected state
            state = None
            if coach_json.exists():
                coach_data = json.loads(coach_json.read_text(encoding="utf-8"))
                state = coach_data.get("corrected_state")
            if not state:
                state = local_data.get("state")
            if not state:
                continue

            # Convert detections to state_classifier format
            det_list = []
            for det in detections:
                obj_type = det.get("type", "")
                cid = EXTENDED_NAME_TO_ID.get(obj_type)
                if cid is None:
                    continue
                det_list.append({"class_id": cid, "confidence": det.get("confidence", 0.5)})

            samples.append({
                "frame": f"{session_dir.name}_{prefix}",
                "state": state,
                "detections": det_list,
            })

    out_path = out_dir / "state_labels.json"
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Extracted %d state labels to %s", len(samples), out_path)
    return len(samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retrain from coach session data")
    parser.add_argument("--sessions", type=Path, required=True, help="Directory containing session subdirs")
    parser.add_argument("--out", type=Path, default=Path("dataset/coach_training"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    sessions = _find_sessions(args.sessions)
    if not sessions:
        logger.error("No sessions found in %s (need session_summary.json in subdirs)", args.sessions)
        return 1
    logger.info("Found %d sessions in %s", len(sessions), args.sessions)

    args.out.mkdir(parents=True, exist_ok=True)
    n_yolo = extract_yolo_data(sessions, args.out, val_ratio=args.val_ratio, seed=args.seed)
    n_state = extract_state_labels(sessions, args.out)

    print("\n===== Coach retrain data extraction =====")
    print(f"Sessions found:       {len(sessions)}")
    print(f"YOLO samples:         {n_yolo}")
    print(f"State label samples:  {n_state}")
    print(f"YOLO dataset:         {args.out / 'yolo'}")
    print(f"State labels:         {args.out / 'state_labels.json'}")
    print("\nNext steps:")
    print(f"  1. Train YOLO:  python -m tools.labeling.finetune_fsm --buttons {args.out / 'yolo'} --base runs/detect/fsm/weights/best.pt --out runs/detect/fsm_v2")
    print(f"  2. Train state: python -m decision.state_classifier --train --data {args.out / 'state_labels.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
