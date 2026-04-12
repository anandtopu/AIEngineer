from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> ConfusionMatrix:
    y_true = y_true.astype(int).ravel()
    y_pred = y_pred.astype(int).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def precision(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fp
    return cm.tp / denom if denom else 0.0


def recall(cm: ConfusionMatrix) -> float:
    denom = cm.tp + cm.fn
    return cm.tp / denom if denom else 0.0


def f1(cm: ConfusionMatrix) -> float:
    p = precision(cm)
    r = recall(cm)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def accuracy(cm: ConfusionMatrix) -> float:
    total = cm.tp + cm.fp + cm.tn + cm.fn
    return (cm.tp + cm.tn) / total if total else 0.0


def main():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])

    cm = confusion_matrix_binary(y_true, y_pred)

    print(cm)
    print(f"accuracy={accuracy(cm):.3f}")
    print(f"precision={precision(cm):.3f}")
    print(f"recall={recall(cm):.3f}")
    print(f"f1={f1(cm):.3f}")


if __name__ == "__main__":
    main()
