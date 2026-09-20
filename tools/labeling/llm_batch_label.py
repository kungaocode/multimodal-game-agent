"""LLM batch labeling: send a directory of images to a vision LLM, produce YOLO labels.

Usage:
    python -m tools.labeling.llm_batch_label --images dataset/frames --out dataset/llm_labeled
    python -m tools.labeling.llm_batch_label --images final_test_dataset --out dataset/llm_labeled --resume

Requires VISION_MODEL_BASE_URL / VISION_MODEL_API_KEY / VISION_MODEL_NAME.
Supports --resume to skip already-labeled images (checks for existing .txt).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from PIL import Image

from perception.auto_labeler import FrameLabel, YoloBox, save_yolo_dataset
from perception.fsm_labels import EXTENDED_FSM_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def collect_images(image_dir: Path) -> list[Path]:
    """Collect image files from directory (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM batch labeling for image directory")
    parser.add_argument("--images", type=Path, required=True, help="Directory of images to label")
    parser.add_argument("--out", type=Path, default=Path("dataset/llm_labeled"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true", help="Skip images that already have labels")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    from perception.vision_model import VisionModel

    vision = VisionModel()
    if not vision.base_url:
        logger.error("VISION_MODEL_BASE_URL not set. Configure env vars first.")
        return 1
    logger.info("LLM labeling with model: %s", vision.model)

    images = collect_images(args.images)
    if not images:
        logger.error("No images found in %s", args.images)
        return 1
    logger.info("Found %d images in %s", len(images), args.images)

    labels: list[FrameLabel] = []
    image_pils: list[Image.Image] = []
    total_boxes = 0
    total_states: dict[str, int] = {}
    skipped = 0
    t0 = time.time()

    out_raw = args.out / "_raw"
    out_raw.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(images):
        # Resume: skip if raw JSON + YOLO label already exist
        stem = img_path.stem
        raw_path = out_raw / f"{stem}.json"
        if args.resume and raw_path.exists():
            skipped += 1
            # Load cached raw output and re-parse
            cached = json.loads(raw_path.read_text(encoding="utf-8"))
            lab = _parse_cached(cached, img_path)
            labels.append(lab)
            image_pils.append(Image.open(img_path).convert("RGB"))
            continue

        lab = FrameLabel(image_path=str(img_path))
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        from perception.fsm_labels import build_llm_label_prompt
        prompt = build_llm_label_prompt(w, h)
        try:
            parsed = vision.chat(img, prompt=prompt)
        except Exception as exc:
            logger.warning("LLM failed for %s: %s", img_path.name, exc)
            parsed = {}

        # Save raw output for resume/audit
        raw_path.write_text(
            json.dumps({"image": img_path.name, "response": parsed}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Parse into YoloBox
        if isinstance(parsed, dict):
            lab.state = parsed.get("state")
            for obj in parsed.get("objects", []):
                if not isinstance(obj, dict):
                    continue
                obj_type = obj.get("type", "")
                from perception.fsm_labels import EXTENDED_NAME_TO_ID
                cid = EXTENDED_NAME_TO_ID.get(obj_type)
                if cid is None:
                    continue
                from perception.auto_labeler import _parse_llm_bbox
                bb = _parse_llm_bbox(obj, w, h)
                if bb is None:
                    continue
                cx, cy, bw, bh = bb
                conf = float(obj.get("confidence") or 0.5)
                lab.boxes.append(YoloBox(class_id=cid, cx=cx, cy=cy, w=bw, h=bh, confidence=conf, source="llm"))

        labels.append(lab)
        image_pils.append(img)
        total_boxes += len(lab.boxes)
        if lab.state:
            total_states[lab.state] = total_states.get(lab.state, 0) + 1

        if (i + 1) % 20 == 0 or i == len(images) - 1:
            elapsed = time.time() - t0
            logger.info(
                "Labeled %d/%d (%.1f img/s, %d boxes, %d skipped)",
                i + 1, len(images), (i + 1) / elapsed if elapsed > 0 else 0, total_boxes, skipped,
            )
        if args.delay > 0 and i < len(images) - 1:
            time.sleep(args.delay)

    # Save YOLO dataset
    out = save_yolo_dataset(
        labels=labels,
        images=image_pils,
        out_dir=args.out,
        val_ratio=args.val_ratio,
        class_names=EXTENDED_FSM_ORDER,
        seed=args.seed,
    )

    print("\n===== LLM batch labeling summary =====")
    print(f"Images:       {len(images)}")
    print(f"Skipped:      {skipped} (resume)")
    print(f"Total boxes:  {total_boxes}")
    print(f"Output:       {out}")
    if total_states:
        print("State distribution:")
        for state, count in sorted(total_states.items(), key=lambda kv: -kv[1]):
            print(f"  {state}: {count} frames")
    return 0


def _parse_cached(cached: dict, img_path: Path) -> FrameLabel:
    """Re-parse a cached LLM output into FrameLabel (for resume)."""
    lab = FrameLabel(image_path=str(img_path))
    parsed = cached.get("response", {})
    if not isinstance(parsed, dict):
        return lab
    lab.state = parsed.get("state")
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    for obj in parsed.get("objects", []):
        if not isinstance(obj, dict):
            continue
        obj_type = obj.get("type", "")
        from perception.fsm_labels import EXTENDED_NAME_TO_ID
        cid = EXTENDED_NAME_TO_ID.get(obj_type)
        if cid is None:
            continue
        from perception.auto_labeler import _parse_llm_bbox
        bb = _parse_llm_bbox(obj, w, h)
        if bb is None:
            continue
        cx, cy, bw, bh = bb
        conf = float(obj.get("confidence") or 0.5)
        lab.boxes.append(YoloBox(class_id=cid, cx=cx, cy=cy, w=bw, h=bh, confidence=conf, source="llm"))
    return lab


if __name__ == "__main__":
    raise SystemExit(main())
