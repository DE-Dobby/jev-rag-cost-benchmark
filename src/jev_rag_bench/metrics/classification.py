from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float  # 환각(1) 클래스 기준
    recall: float
    f1: float
    n: int


def classification_metrics(preds: list[bool], labels: list[bool]) -> ClassificationMetrics:
    if len(preds) != len(labels):
        raise ValueError("preds와 labels 길이가 다릅니다")
    n = len(preds)
    if n == 0:
        return ClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, n=0)

    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    correct = sum(1 for p, y in zip(preds, labels) if p == y)

    accuracy = correct / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return ClassificationMetrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, n=n)
