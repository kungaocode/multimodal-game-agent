"""Self-training loop: train -> pseudo-label -> filter high-conf -> retrain.

Usage:
    # Round 1: train on existing dataset, pseudo-label unlabeled frames
    python -m tools.labeling.self_train --dataset dataset/video_labeled --unlabeled dataset/raw_frames --rounds 3

    # Custom thresholds:
    python -m tools.labeling.self_train --dataset dataset/video_labeled --unlabeled dataset/raw_frames --rounds 2 --conf 0.85 --llm-conf 0.4

Pipeline per round:
    1. Train YOLO on current dataset (train + pseudo from previous round)
    2. Run model on unlabeled frames
    3. conf >= --conf    -> add as pseudo-label (model output)
    4. conf <  --llm-conf -> send to LLM for labeling (optional, needs env vars)
    5. Merge pseudo + LLM labels into dataset, go to next round

Each round produces: runs/self_train/round_N/weights/best.pt
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from perception.auto_labeler import YoloBox, FrameLabel, save_yolo_dataset
from perception.detector import Detector
from perception.fsm_labels import EXTENDED_FSM_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_unlabeled(image_dir: Path, max_images: int = 500) -> list[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)[:max_images]
    return [Image.open(p).convert("RGB") for p in paths]


def _pseudo_label(
    detector: Detector,
    images: list[Image.Image],
    conf_threshold: float,
    llm_conf_threshold: float,
    vision: object | None = None,
) -> tuple[list[FrameLabel], list[Image.Image], int]:
    """Run detector on images, split into pseudo-label (high conf) and LLM-needed (low conf)."""
    pseudo_labels: list[FrameLabel] = []
    pseudo_images: list[Image.Image] = []
    llm_needed: list[Image.Image] = []
    for i, img in enumerate(images):
        detections = detector.predict(img, conf=llm_conf_threshold)
        lab = FrameLabel(image_path=f"<unlabeled_{i}>")
        high_conf_boxes: list[YoloBox] = []
        low_conf_boxes: list[YoloBox] = []
        for det in detections:
            box = YoloBox(
                class_id=det.class_id,
                cx=(det.bbox[0] + det.bbox[2]) / 2 / img.width,
                cy=(det.bbox[1] + det.bbox[3]) / 2 / img.height,
                w=(det.bbox[2] - det.bbox[0]) / img.width,
                h=(det.bbox[3] - det.bbox[1]) / img.height,
                confidence=det.confidence,
                source="pseudo",
            )
            if det.confidence >= conf_threshold:
                high_conf_boxes.append(box)
            else:
                low_conf_boxes.append(box)
        if high_conf_boxes:
            lab.boxes = high_conf_boxes
            pseudo_labels.append(lab)
            pseudo_images.append(img)
        elif low_conf_boxes and vision is not None:
            # Low confidence -> send to LLM for correction
            llm_needed.append(img)
        elif not detections:
            # No detections at all: keep as negative sample
            pseudo_labels.append(lab)
            pseudo_images.append(img)
    return pseudo_labels, pseudo_images, len(llm_needed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-training loop")
    parser.add_argument("--dataset", type=Path, required=True, help="Initial YOLO dataset dir")
    parser.add_argument("--unlabeled", type=Path, required=True, help="Directory of unlabeled frames")
    parser.add_argument("--rounds", type=int, default=3, help="Number of self-training rounds")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs per round")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.80, help="Pseudo-label confidence threshold")
    parser.add_argument("--llm-conf", type=float, default=0.40, help="Below this -> send to LLM")
    parser.add_argument("--llm", action="store_true", help="Enable LLM for low-conf frames")
    parser.add_argument("--max-unlabeled", type=int, default=500, help="Max unlabeled images per round")
    parser.add_argument("--out", type=Path, default=Path("runs/self_train"))
    parser.add_argument("--base", type=str, default=None, help="Base weights (default: dataset's last or yolov8n)")
    args = parser.parse_args(argv)

    vision = None
    if args.llm:
        try:
            from perception.vision_model import VisionModel
            vision = VisionModel()
            if not vision.base_url:
                vision = None
        except Exception:
            vision = None

    # Determine base weights
    base = args.base
    if base is None:
        # Try existing trained FSM weights first
        fsm_best = Path("runs/detect/fsm/weights/best.pt")
        if fsm_best.exists():
            base = str(fsm_best)
        else:
            base = "yolov8n.pt"
    logger.info("Base weights: %s", base)

    # Load initial dataset images
    train_img_dir = args.dataset / "images" / "train"
    train_lab_dir = args.dataset / "labels" / "train"
    if not train_img_dir.exists():
        logger.error("Dataset not found at %s", args.dataset)
        return 1

    dataset_yaml = args.dataset / "dataset.yaml"
    if not dataset_yaml.exists():
        logger.error("dataset.yaml not found in %s", args.dataset)
        return 1

    unlabeled_images = _load_unlabeled(args.unlabeled, args.max_unlabeled)
    logger.info("Loaded %d unlabeled images from %s", len(unlabeled_images), args.unlabeled)

    current_base = base
    for round_num in range(1, args.rounds + 1):
        logger.info("===== Round %d / %d =====", round_num, args.rounds)
        round_out = args.out / f"round_{round_num}"
        round_out.mkdir(parents=True, exist_ok=True)

        # 1. Train
        detector = Detector(current_base)
        logger.info("Training round %d (epochs=%d, base=%s)", round_num, args.epochs, current_base)
        best_weights = detector.train(
            data_yaml=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(args.out),
            name=f"round_{round_num}",
            exist_ok=True,
        )
        logger.info("Round %d best weights: %s", round_num, best_weights)

        # 2. Pseudo-label unlabeled frames
        if round_num < args.rounds:
            det_for_pseudo = Detector(str(best_weights))
            pseudo_labels, pseudo_images, n_llm = _pseudo_label(
                det_for_pseudo, unlabeled_images, args.conf, args.llm_conf, vision
            )
            n_pseudo = len(pseudo_labels)
            logger.info(
                "Round %d pseudo-labeling: %d high-conf, %d need LLM, %d negatives",
                round_num, n_pseudo, n_lln if False else n_llm, n_pseudo,
            )

            # 3. Save pseudo-labeled data and merge into dataset
            if pseudo_images:
                pseudo_dir = round_out / "pseudo_dataset"
                save_yolo_dataset(
                    labels=pseudo_labels,
                    images=pseudo_images,
                    out_dir=pseudo_dir,
                    val_ratio=0.0,  # all go to train
                    class_names=EXTENDED_FSM_ORDER,
                )
                # Copy pseudo images+labels into main dataset's train set
                _merge_pseudo_into_dataset(pseudo_dir, args.dataset)
                logger.info("Merged %d pseudo-labeled images into %s", n_pseudo, args.dataset)

            # 4. Update base for next round
            current_base = str(best_weights)

    logger.info("Self-training complete. Final weights: %s", current_base)
    print(f"\nFinal weights: {current_base}")
    return 0


def _merge_pseudo_into_dataset(pseudo_dir: Path, dataset_dir: Path) -> None:
    """Copy pseudo-labeled images+labels from pseudo_dir into dataset_dir's train set."""
    pseudo_img_dir = pseudo_dir / "images" / "train"
    pseudo_lab_dir = pseudo_dir / "labels" / "train"
    if not pseudo_img_dir.exists():
        return
    dst_img_dir = dataset_dir / "images" / "train"
    dst_lab_dir = dataset_dir / "labels" / "train"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)
    for img in sorted(pseudo_img_dir.glob("*.jpg")):
        # Prefix with "pseudo_" to avoid name collision
        dst_img = dst_img_dir / f"pseudo_{img.name}"
        dst_lab = dst_lab_dir / f"pseudo_{img.stem}.txt"
        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        lab = pseudo_lab_dir / f"{img.stem}.txt"
        if lab.exists() and not dst_lab.exists():
            shutil.copy2(lab, dst_lab)
