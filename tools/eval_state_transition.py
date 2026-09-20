"""状态转换识别测试（阶段 13 实验评估体系 · 打资源 FSM 感知验收）。

用户目标：验证「训练完成的本地模型」能否识别出状态转换标志——
从截图识别状态信号（进攻按钮/搜索对手/下一个/结束战斗/返回按钮/敌方资源建筑），
驱动 FSM 完成 村庄待机 → 搜索中 → 战斗中 → 战斗结束 → 村庄待机 的流转。

本工具对一组截图（默认 final_test_dataset/）：
  1. 跑本地检测器（farm 8 类 / fsm 12 类权重均可），输出每个状态信号；
  2. 可选跑 OCR 作为「参考信号」（文本识别的按钮位置，用于对照模型能力差距）；
  3. 对照 ground truth（labels.json，期望状态 + 期望信号）计算：
       - 状态识别准确率（信号规则判定）
       - 各状态信号的检出率（recall）
       - 资源建筑检出情况
  4. --sequence 模式：按序列顺序喂给 BattleFSM，验证完整状态转换链。

参数：
    --images   图片目录（默认 final_test_dataset）
    --labels   ground truth json（默认 <images>/labels.json）
    --weights  检测器权重（默认 runs/detect/farm/weights/best.pt）
    --ocr      同时跑 OCR 参考信号
    --sequence 逗号分隔的文件名序列，驱动 BattleFSM 走完整转换
    --json     输出结构化报告到文件

用法：
    python tools/eval_state_transition.py --images final_test_dataset --ocr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许从 tools/ 直接运行（python tools/eval_state_transition.py）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from decision.battle_fsm import BattleFSM, BattleSignals
from perception.detector import Detector
from perception.fsm_labels import class_id_for_button_text, id_for_name, name_for_id
from perception.ocr import OCRReader

RESOURCE_NAMES = {"圣水收集器", "金矿", "圣水瓶", "储金罐", "暗黑重油罐", "暗黑重油钻井"}

# 按钮检测类名 → BattleSignals 字段
BUTTON_SIGNAL_FIELD = {
    "进攻按钮": "attack_button",
    "返回按钮": "return_button",
    "搜索对手按钮": "search_button",
    "下一个按钮": "next_button",
    "结束战斗按钮": "end_battle_button",
}

# 期望状态 → 判定规则优先级（信号规则判定）
STATE_RULES = [
    ("战斗中", lambda s: s.end_battle_button is not None or s.next_button is not None),
    ("村庄待机", lambda s: s.attack_button is not None and s.end_battle_button is None),
    ("搜索中", lambda s: s.search_button is not None or s.return_button is not None),
]
DEFAULT_STATE = "未知/等待"


def load_detector(weights: str | Path | None):
    if weights is None:
        return None
    try:
        return Detector(str(weights))
    except FileNotFoundError:
        return None


def yolo_signals(det, image: Image.Image) -> BattleSignals:
    """检测器输出 → BattleSignals。"""
    signals = BattleSignals()
    if det is None:
        return signals
    dets = det.predict(image, conf=0.25)
    by_name: dict[str, list] = {}
    for d in dets:
        by_name.setdefault(d.class_name, []).append(d)
    for name, field_name in BUTTON_SIGNAL_FIELD.items():
        items = by_name.get(name, [])
        if items:
            best = max(items, key=lambda d: d.confidence)
            cx = (best.bbox[0] + best.bbox[2]) // 2
            cy = (best.bbox[1] + best.bbox[3]) // 2
            setattr(signals, field_name, (cx, cy))
    for name in RESOURCE_NAMES:
        for d in by_name.get(name, []):
            cx = (d.bbox[0] + d.bbox[2]) // 2
            cy = (d.bbox[1] + d.bbox[3]) // 2
            signals.enemy_resources.append((d.class_name, (cx, cy)))
    return signals


def ocr_signals(image: Image.Image | Path, ocr: OCRReader | None) -> BattleSignals:
    """OCR 文本 → 参考信号（模型能力对照）。"""
    signals = BattleSignals()
    if ocr is None:
        return signals
    for r in ocr.read_text(image):
        cid = class_id_for_button_text(r.text)
        if cid is None:
            continue
        name = name_for_id(cid)
        field_name = BUTTON_SIGNAL_FIELD.get(name)
        if field_name is None:
            continue
        xs = [p[0] for p in r.bbox]
        ys = [p[1] for p in r.bbox]
        cx, cy = int((min(xs) + max(xs)) / 2), int((min(ys) + max(ys)) / 2)
        setattr(signals, field_name, (cx, cy))
    return signals


def signal_rules_state(signals: BattleSignals) -> str:
    """用信号规则判定界面状态（用于单帧对比）。"""
    for state, rule in STATE_RULES:
        if rule(signals):
            return state
    return DEFAULT_STATE


def signal_fields(signals: BattleSignals) -> dict[str, bool]:
    return {
        "attack_button": signals.attack_button is not None,
        "return_button": signals.return_button is not None,
        "search_button": signals.search_button is not None,
        "next_button": signals.next_button is not None,
        "end_battle_button": signals.end_battle_button is not None,
    }


def _pick_signals(yolo: BattleSignals, ocr: BattleSignals | None) -> BattleSignals:
    """优先 YOLO 信号；YOLO 全空时用 OCR 兜底（便于无权重时预览 FSM 链路）。"""
    if (
        yolo.attack_button or yolo.return_button or yolo.search_button
        or yolo.next_button or yolo.end_battle_button or yolo.enemy_resources
    ):
        return yolo
    if ocr is not None:
        return ocr
    return yolo


def run_eval(
    images_dir: Path,
    labels_path: Path,
    weights: str | Path | None,
    use_ocr: bool,
    sequence: list[str] | None,
) -> dict:
    det = load_detector(weights)
    ocr = OCRReader(lang="ch") if use_ocr else None
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {"samples": {}}
    samples = labels.get("samples", {})

    per_image = []
    signal_tp: dict[str, int] = {}
    signal_exp: dict[str, int] = {}
    state_correct = state_total = 0
    donation_total = 0
    res_detected = res_total = 0

    image_paths = sorted(Path(images_dir).glob("*.jpg")) + sorted(Path(images_dir).glob("*.png"))
    for img_path in image_paths:
        meta = samples.get(img_path.name, {})
        exp_state = meta.get("expected_state", "")
        exp_signals = meta.get("signals", {})
        img = Image.open(img_path).convert("RGB")

        sig_yolo = yolo_signals(det, img) if det else BattleSignals()
        sig_ocr = ocr_signals(img, ocr) if ocr else None
        sig = _pick_signals(sig_yolo, sig_ocr)
        pred_state = signal_rules_state(sig)

        # 信号检出统计
        for field, expect in exp_signals.items():
            if expect is not True:
                continue
            if field in signal_fields(sig):
                signal_exp[field] = signal_exp.get(field, 0) + 1
                if getattr(sig, field) is not None:
                    signal_tp[field] = signal_tp.get(field, 0) + 1
        # 状态统计
        if exp_state:
            if "捐兵" in exp_state:
                donation_total += 1
            else:
                state_total += 1
                if pred_state == exp_state:
                    state_correct += 1
        # 资源建筑统计
        if exp_state in ("战斗中", "搜索中", "村庄待机"):
            res_total += 1
            if sig.enemy_resources:
                res_detected += 1

        per_image.append(
            {
                "file": img_path.name,
                "expected_state": exp_state,
                "predicted_state": pred_state,
                "state_match": bool(exp_state and pred_state == exp_state),
                "donation_scene": "捐兵" in exp_state,
                "res_detected": len(sig.enemy_resources),
                "yolo_signals": signal_fields(sig_yolo),
                "ocr_signals": signal_fields(sig_ocr) if sig_ocr else None,
                "res_names": sorted({n for n, _ in sig.enemy_resources}),
            }
        )

    sequence_report = run_sequence(images_dir, sequence, det, ocr) if sequence else None
    return {
        "images_dir": str(images_dir),
        "labels": str(labels_path),
        "weights": str(weights) if weights else None,
        "state_accuracy": {
            "correct": state_correct,
            "total": state_total,
            "rate": round(state_correct / state_total, 4) if state_total else None,
        },
        "donation_scenes": donation_total,
        "signal_recall": {
            k: {
                "tp": signal_tp.get(k, 0),
                "expected": signal_exp.get(k, 0),
                "rate": round(signal_tp.get(k, 0) / signal_exp[k], 4) if signal_exp.get(k) else None,
            }
            for k in ["attack_button", "return_button", "search_button", "next_button", "end_battle_button"]
        },
        "resource_event": {"frames_with_resources": res_detected, "frames_checked": res_total},
        "yolo_independent": {
            "frames_checked": len(per_image),
            "frames_with_yolo_buttons": sum(1 for it in per_image if any(it["yolo_signals"].values())),
            "total_yolo_signals": sum(sum(it["yolo_signals"].values()) for it in per_image),
            "tp_yolo": sum(
                1 for it in per_image
                for fld, exp in (labels.get("samples", {}).get(it["file"], {}).get("signals", {})).items()
                if exp is True and it["yolo_signals"].get(fld)
            ),
        },
        "per_image": per_image,
        "sequence": sequence_report,
    }


def run_sequence(images_dir: Path, sequence: list[str], det, ocr=None) -> dict:
    """按序列图片依次驱动 BattleFSM，验证完整状态转换链。"""
    fsm = BattleFSM()
    steps = []
    for name in sequence:
        path = images_dir / name
        if not path.exists():
            steps.append({"file": name, "error": "文件不存在"})
            continue
        img = Image.open(path).convert("RGB")
        sig_yolo = yolo_signals(det, img) if det else BattleSignals()
        sig_ocr = ocr_signals(img, ocr) if ocr else None
        sig = _pick_signals(sig_yolo, sig_ocr)
        action = fsm.step(sig)
        steps.append({"file": name, "action": f"{action.kind}:{action.target}", "fsm_state": fsm.state.value})
    return {"history": fsm.history, "steps": steps}


def print_report(report: dict) -> None:
    print("=" * 76)
    print("状态转换识别测试报告")
    print(f"  图片目录: {report['images_dir']}")
    print(f"  权重:     {report['weights'] or '(未指定，仅 OCR 参考)'}")
    print("=" * 76)
    print("\n[逐图识别]（期望状态 vs 信号规则判定）")
    print(f"{'文件':<46}{'期望状态':<9}{'判定状态':<9}{'结果'}")
    for it in report["per_image"]:
        mark = "✅" if it["state_match"] else ("(捐兵)" if it["donation_scene"] else "❌")
        res = it["res_detected"]
        print(f"{it['file']:<46}{it['expected_state']:<9}{it['predicted_state']:<9}{mark}  资源{res}个")
    sa = report["state_accuracy"]
    print("\n[状态识别准确率]（打资源 FSM 场景，不含捐兵）")
    if sa["total"]:
        print(f"  正确 {sa['correct']}/{sa['total']} = {sa['rate'] * 100:.1f}%"
              f"  （捐兵场景 {report['donation_scenes']} 张，属捐兵 FSM 范围外）")
    else:
        print("  无打资源 FSM 场景样本")
    print("\n[按钮信号检出率]（含 OCR 兜底，vs ground truth 期望）")
    for k, v in report["signal_recall"].items():
        rate = f"{v['rate'] * 100:.0f}%" if v["rate"] is not None else "—"
        print(f"  {k:<20} {v['tp']}/{v['expected']} ({rate})")
    yolo = report.get("yolo_independent")
    if yolo is not None:
        print(f"\n[YOLO 独立检出]（不加 OCR，模型真实能力）")
        print(f"  有按钮检出的帧: {yolo['frames_with_yolo_buttons']}/{yolo['frames_checked']}")
        print(f"  按钮信号数(YOLO): {yolo['total_yolo_signals']}  其中命中期望: {yolo['tp_yolo']}")
    re_ = report["resource_event"]
    print(f"\n[资源建筑] 检出帧 {re_['frames_with_resources']}/{re_['frames_checked']}")
    seq = report.get("sequence")
    if seq:
        print(f"\n[状态转换序列] {seq['history']}")
        for s in seq["steps"]:
            print(f"  {s.get('file')}: {s.get('action')} -> {s.get('fsm_state')}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="状态转换识别测试")
    parser.add_argument("--images", default="final_test_dataset", type=Path)
    parser.add_argument("--labels", default=None, type=Path, help="ground truth json")
    parser.add_argument("--weights", default="runs/detect/farm/weights/best.pt", type=str)
    parser.add_argument("--ocr", action="store_true", help="同时跑 OCR 作为参考信号")
    parser.add_argument("--sequence", default=None, help="逗号分隔文件名序列，驱动 BattleFSM")
    parser.add_argument("--json", default=None, type=Path, help="结构化报告输出路径")
    parser.add_argument("--fatal", action="store_true", help="状态准确率不足 100% 时返回非零")
    args = parser.parse_args(argv)

    labels_path = args.labels or args.images / "labels.json"
    if not labels_path.exists():
        print(f"未找到 ground truth: {labels_path}", file=sys.stderr)
        return 2
    seq = [s.strip() for s in args.sequence.split(",")] if args.sequence else None

    report = run_eval(
        images_dir=args.images,
        labels_path=labels_path,
        weights=args.weights,
        use_ocr=args.ocr,
        sequence=seq,
    )
    print_report(report)
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结构化报告已写入: {args.json}")
    sa = report["state_accuracy"]
    if args.fatal and sa["total"] and sa["correct"] < sa["total"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
