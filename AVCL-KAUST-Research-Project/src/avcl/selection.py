from __future__ import annotations

import numpy as np


def uniform_indices(n_frames: int, budget: int) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, budget, dtype=int).tolist()


def random_indices(
    n_frames: int,
    budget: int,
    rng: np.random.Generator | None = None,
) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    rng = rng if rng is not None else np.random.default_rng(42)
    return sorted(rng.choice(n_frames, size=budget, replace=False).tolist())


def adaptive_indices(
    frames: list[np.ndarray],
    budget: int,
    motion_weight: float = 0.4,
    diversity_weight: float = 0.4,
    coverage_weight: float = 0.2,
) -> list[int]:
    """Greedily select frames using motion, visual novelty, and temporal coverage."""
    n = len(frames)
    if n <= 0 or budget <= 0:
        return []
    if n <= budget:
        return list(range(n))

    weights = np.asarray(
        [motion_weight, diversity_weight, coverage_weight], dtype=np.float32
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, and sum to > 0.")
    weights /= weights.sum()

    gray = [
        np.asarray(frame, dtype=np.float32).mean(axis=2) / 255.0
        for frame in frames
    ]
    motion = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        motion[i] = np.mean(np.abs(gray[i] - gray[i - 1]))
    if motion.max() > 0:
        motion /= motion.max()

    descriptors = np.stack([frame.reshape(-1) for frame in gray])
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(norms, 1e-8)

    selected = [0, n - 1] if budget >= 2 else [0]
    while len(selected) < budget:
        candidates = [i for i in range(n) if i not in selected]
        scores = []
        for i in candidates:
            diversity = min(
                float(np.linalg.norm(descriptors[i] - descriptors[j]))
                for j in selected
            )
            temporal = min(abs(i - j) for j in selected) / max(n - 1, 1)
            scores.append(
                weights[0] * float(motion[i])
                + weights[1] * diversity
                + weights[2] * temporal
            )
        selected.append(candidates[int(np.argmax(scores))])
    return sorted(selected)
