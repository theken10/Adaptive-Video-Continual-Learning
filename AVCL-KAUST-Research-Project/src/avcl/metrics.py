from __future__ import annotations
import numpy as np

def average_accuracy(matrix: np.ndarray, trained_task: int) -> float:
    values = matrix[trained_task, :trained_task + 1]
    values = values[~np.isnan(values)]
    return float(values.mean()) if len(values) else float("nan")

def forgetting(matrix: np.ndarray, trained_task: int) -> float:
    values = []
    for task in range(trained_task):
        history = matrix[task:trained_task + 1, task]
        history = history[~np.isnan(history)]
        current = matrix[trained_task, task]
        if len(history) and not np.isnan(current):
            values.append(float(history.max() - current))
    return float(np.mean(values)) if values else 0.0

def backward_transfer(matrix: np.ndarray, trained_task: int) -> float:
    values = []
    for task in range(trained_task):
        initial, current = matrix[task, task], matrix[trained_task, task]
        if not np.isnan(initial) and not np.isnan(current):
            values.append(float(current - initial))
    return float(np.mean(values)) if values else 0.0
