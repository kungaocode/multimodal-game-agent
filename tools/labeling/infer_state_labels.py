"""Build state labels from a YOLO dataset for legacy auto-labeled outputs.

This is a deterministic fallback for datasets created before ``state_labels.json``
was persisted by ``save_yolo_dataset``.  It uses strong UI signals to label the
dominant state; unlabeled frames are kept with ``state=null``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from perception.fsm_labels import EXTENDED_FSM_ORDER, EXTENDED_NAME_TO_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESOURCE_CLASSES = {
    "金矿", "圣水收集器", "储金罐", "圣水瓶", "暗黑重油罐", "暗黑重油钻井",
}


def infer_state(class_names: set[str]) -> str | None:
    """Infer the game state from high-signal classes."""
    if "请求条目" in class_names:
        return "捐兵-消息列表"
    if "捐赠确认按钮" in class_names:
        return "捐兵-增援确认"
    if "下一个按钮" in class_names or "搜索对手按钮" in class_names:
        return "搜索中"
    if "返回按钮" in class_names:
        return "战斗结束"
    if "结束战斗按钮" in class_names or (class_names & RESOURCE_CLASSES):
        return "战斗中"
    if "进攻按钮" in class_names or "增援按钮" in class_names:
        return "村庄待机"
    return None


def collect_samples(root: Path, default_confidence: float = 0.85) -> list[dict]:
    label_map = json.loads((root / "labels.json").read_text(encoding="utf-8"))
    id_to_name = {int(k): v for k, v in label_map.items()}
    samples: list[dict] = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            class_ids: list[int] = []
            if label_path.exists():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    fields = line.strip().split()
                    if fields:
                        class_ids.append(int(float(fields[0])))
            class_names = {
                id_to_name[cid]
                for cid in class_ids
                if 0 <= cid < len(id_to_name) and id_to_name[cid] in EXTENDED_NAME_TO_ID
            }
            samples.append({
                "frame": f"images/{split}/{image_path.name}",
                "state": infer_state(class_names),
                "detections": [
                    {"class_id": cid, "confidence": default_confidence}
                    for cid in class_ids
                    if 0 <= cid < len(EXTENDED_FSM_ORDER)
                ],
            })
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Infer state labels from YOLO classes")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to <dataset>/state_labels.json")
    parser.add_argument("--confidence", type=float, default=0.85)
    args = parser.parse_args(argv)

    samples = collect_samples(args.dataset, args.confidence)
    out_path = args.out or args.dataset / "state_labels.json"
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = sum(1 for sample in samples if sample["state"])
    logger.info("Wrote %d state labels (%d valid) to %s", len(samples), valid, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
