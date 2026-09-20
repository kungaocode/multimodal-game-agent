"""Learned state classifier: detection signals -> FSM state prediction.

Instead of hand-coding signal->state rules, this MLP learns the mapping from
perception output (which objects are visible + confidence scores) to game state.
Trained on LLM-labeled video frames (each frame has a "state" annotation).

Usage:
    # Train:
    python -m decision.state_classifier --train --data dataset/llm_labeled/state_labels.json --out runs/state_classifier

    # Predict:
    from decision.state_classifier import StateClassifier
    clf = StateClassifier.load("runs/state_classifier/best.pt")
    state = clf.predict(detection_results)  # list of DetectionResult
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from perception.fsm_labels import EXTENDED_FSM_ORDER

logger = logging.getLogger(__name__)

# Game states (must match LLM labeling prompt's "state" values)
GAME_STATES = [
    "村庄待机",
    "搜索中",
    "战斗中",
    "战斗结束",
    "捐兵-消息列表",
    "捐兵-增援确认",
]
STATE_TO_IDX = {s: i for i, s in enumerate(GAME_STATES)}
IDX_TO_STATE = {i: s for i, s in enumerate(GAME_STATES)}


def signals_to_features(
    detections: list,  # list of DetectionResult
    num_classes: int | None = None,
) -> np.ndarray:
    """Convert detection results to a fixed-length feature vector.

    For each class: [max_confidence, count] -> feature dim = num_classes * 2.
    """
    if num_classes is None:
        num_classes = len(EXTENDED_FSM_ORDER)
    max_conf = np.zeros(num_classes, dtype=np.float32)
    counts = np.zeros(num_classes, dtype=np.float32)
    for det in detections:
        if isinstance(det, dict):
            cid = det.get("class_id")
            confidence = det.get("confidence", 0.5)
        else:
            cid = det.class_id
            confidence = det.confidence
        if cid is None:
            continue
        cid = int(cid)
        if 0 <= cid < num_classes:
            max_conf[cid] = max(max_conf[cid], float(confidence))
            counts[cid] += 1
    return np.concatenate([max_conf, counts])


class StateClassifier:
    """Lightweight MLP for state classification from detection signals."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_states: int = 6) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        self._model = None

    def _build(self) -> object:
        import torch
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.num_states),
        )

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> dict:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        n = len(features)
        indices = np.random.RandomState(seed).permutation(n)
        n_val = max(1, int(n * val_ratio)) if n > 1 else 0
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        x_train = torch.tensor(features[train_idx], dtype=torch.float32)
        y_train = torch.tensor(labels[train_idx], dtype=torch.long)
        x_val = torch.tensor(features[val_idx], dtype=torch.float32) if n_val > 0 else None
        y_val = torch.tensor(labels[val_idx], dtype=torch.long) if n_val > 0 else None

        model = self._build()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_state = None
        history = {"train_loss": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(x_train))
            total_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                batch_idx = perm[i : i + batch_size]
                xb, yb = x_train[batch_idx], y_train[batch_idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, (len(x_train) + batch_size - 1) // batch_size)
            history["train_loss"].append(avg_loss)

            if x_val is not None:
                model.eval()
                with torch.no_grad():
                    logits = model(x_val)
                    preds = logits.argmax(dim=1)
                    acc = (preds == y_val).float().mean().item()
                history["val_acc"].append(acc)
                if acc > best_acc:
                    best_acc = acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if (epoch + 1) % 20 == 0:
                    logger.info("Epoch %d: loss=%.4f val_acc=%.3f", epoch + 1, avg_loss, acc)

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        return {"best_val_acc": best_acc, "history": history}

    def predict(self, detections: list) -> str:
        """Predict game state from detection results."""
        import torch

        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")
        feat = signals_to_features(detections, num_classes=self.input_dim // 2)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)
            pred_idx = logits.argmax(dim=1).item()
        return IDX_TO_STATE.get(pred_idx, "未知")

    def predict_proba(self, detections: list) -> dict[str, float]:
        import torch

        if self._model is None:
            raise RuntimeError("Model not trained or loaded.")
        feat = signals_to_features(detections, num_classes=self.input_dim // 2)
        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        self._model.eval()
        with torch.no_grad():
            probs = torch.softmax(self._model(x), dim=1).squeeze(0).numpy()
        return {IDX_TO_STATE.get(i, f"state_{i}"): float(p) for i, p in enumerate(probs)}

    def save(self, path: Path | str) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._model.state_dict() if self._model else None,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_states": self.num_states,
                "game_states": GAME_STATES,
                "class_names": EXTENDED_FSM_ORDER,
            },
            path,
        )
        logger.info("Saved state classifier to %s", path)

    @classmethod
    def load(cls, path: Path | str) -> "StateClassifier":
        import torch

        path = Path(path)
        data = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(
            input_dim=data["input_dim"],
            hidden_dim=data["hidden_dim"],
            num_states=data["num_states"],
        )
        if data.get("model_state_dict"):
            clf._model = clf._build()
            clf._model.load_state_dict(data["model_state_dict"])
            clf._model.eval()
        return clf


def prepare_training_data(
    state_labels_path: Path | str,
    num_classes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare (features, labels) from LLM-labeled frame data.

    Expects a JSON file: [{"frame": "...", "state": "村庄待机", "detections": [{"class_id": ..., "confidence": ...}, ...]}, ...]
    Or: generated from llm_batch_label output (state_labels.json produced separately).
    """
    if num_classes is None:
        num_classes = len(EXTENDED_FSM_ORDER)
    data = json.loads(Path(state_labels_path).read_text(encoding="utf-8"))
    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    for item in data:
        state = item.get("state", "")
        if state not in STATE_TO_IDX:
            continue
        dets = item.get("detections", [])
        feat = signals_to_features(dets, num_classes=num_classes)
        features_list.append(feat)
        labels_list.append(STATE_TO_IDX[state])
    if not features_list:
        raise ValueError(f"No valid training samples in {state_labels_path}")
    return np.stack(features_list), np.array(labels_list, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train state classifier")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data", type=Path, required=True, help="state_labels.json path")
    parser.add_argument("--out", type=Path, default=Path("runs/state_classifier"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args(argv)

    if not args.train:
        parser.print_help()
        return 0

    features, labels = prepare_training_data(args.data)
    logger.info("Training data: %d samples, %d classes", len(features), len(set(labels)))

    clf = StateClassifier(
        input_dim=features.shape[1],
        hidden_dim=args.hidden,
        num_states=len(GAME_STATES),
    )
    result = clf.train(features, labels, epochs=args.epochs, lr=args.lr)
    logger.info("Best val accuracy: %.3f", result["best_val_acc"])

    # Both directory-style and file-style output paths are accepted so that a
    # retrain request ending in .pt does not accidentally create best.pt/best.pt.
    output_path = args.out if args.out.suffix == ".pt" else args.out / "best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(output_path)
    (output_path.parent / "training_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nState classifier saved to {output_path}")
    print(f"Best val accuracy: {result['best_val_acc']:.3f}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
