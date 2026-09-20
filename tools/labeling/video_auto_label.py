"""Video -> frames -> dedup -> OCR + LLM auto-label -> YOLO dataset.

Usage:
    # OCR only (no API cost):
    python -m tools.labeling.video_auto_label --video gameplay.mp4 --out dataset/video_labeled

    # OCR + LLM (needs VISION_MODEL_* env vars):
    python -m tools.labeling.video_auto_label --video gameplay.mp4 --out dataset/video_labeled --llm

    # Multiple videos:
    python -m tools.labeling.video_auto_label --video a.mp4 b.mp4 --out dataset/video_labeled --llm

    # Adjust sampling:
    python -m tools.labeling.video_auto_label --video a.mp4 --fps 0.5 --diff 12 --max-frames 500
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from PIL import Image

from perception.auto_labeler import (
    FrameLabel,
    auto_label_frame,
    extract_frames,
    save_yolo_dataset,
)
from perception.fsm_labels import EXTENDED_FSM_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Video auto-label to YOLO dataset")
    parser.add_argument("--video", nargs="+", type=Path, required=True, help="Video file(s)")
    parser.add_argument("--out", type=Path, default=Path("dataset/video_labeled"))
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to sample")
    parser.add_argument("--diff", type=float, default=8.0, help="Frame diff threshold for dedup")
    parser.add_argument("--max-frames", type=int, default=2000, help="Max frames per video")
    parser.add_argument("--llm", action="store_true", help="Enable LLM batch labeling (requires VISION_MODEL_*)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR labeling")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--ocr-pad", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    all_frames: list[Image.Image] = []
    for video in args.video:
        logger.info("Extracting frames from %s ...", video)
        frames = extract_frames(video, fps=args.fps, diff_threshold=args.diff, max_frames=args.max_frames)
        all_frames.extend(frames)
        logger.info("  -> %d frames (total so far: %d)", len(frames), len(all_frames))

    if not all_frames:
        logger.error("No frames extracted from any video.")
        return 1

    # Setup OCR
    ocr = None
    if not args.no_ocr:
        try:
            from perception.ocr import OCRReader
            ocr = OCRReader()
            logger.info("OCR initialized.")
        except Exception as exc:
            logger.warning("OCR unavailable (%s), skipping OCR labeling.", exc)

    # Setup LLM
    vision = None
    if args.llm:
        try:
            from perception.vision_model import VisionModel
            vision = VisionModel()
            if not vision.base_url:
                logger.warning("VISION_MODEL_BASE_URL not set; LLM labeling disabled.")
                vision = None
            else:
                logger.info("LLM labeling enabled: %s", vision.model)
        except Exception as exc:
            logger.warning("VisionModel unavailable (%s); LLM labeling disabled.", exc)
            vision = None

    if ocr is None and vision is None:
        logger.error("Neither OCR nor LLM available. Nothing to label.")
        return 1

    # Auto-label each frame
    labels: list[FrameLabel] = []
    t0 = time.time()
    for i, frame in enumerate(all_frames):
        lab = auto_label_frame(frame, ocr=ocr, vision=vision, ocr_pad=args.ocr_pad)
        labels.append(lab)
        if (i + 1) % 50 == 0 or i == len(all_frames) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Labeled %d/%d frames (%.1f fps, boxes: %d)",
                i + 1, len(all_frames), rate, sum(l.total_boxes for l in labels),
            )

    # Save dataset
    out = save_yolo_dataset(
        labels=labels,
        images=all_frames,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        class_names=EXTENDED_FSM_ORDER,
        seed=args.seed,
    )

    # Summary
    total_boxes = sum(l.total_boxes for l in labels)
    ocr_total = sum(l.ocr_boxes for l in labels)
    llm_total = sum(l.llm_boxes for l in labels)
    states = {}
    for l in labels:
        if l.state:
            states[l.state] = states.get(l.state, 0) + 1
    empty = sum(1 for l in labels if l.total_boxes == 0)
    print("\n===== Auto-label summary =====")
    print(f"Videos:         {len(args.video)}")
    print(f"Total frames:   {len(all_frames)}")
    print(f"Total boxes:    {total_boxes} (OCR: {ocr_total}, LLM: {llm_total})")
    print(f"Empty frames:   {empty} (no labels, kept as negatives)")
    print(f"Output:         {out}")
    if states:
        print("State distribution:")
        for state, count in sorted(states.items(), key=lambda kv: -kv[1]):
            print(f"  {state}: {count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
