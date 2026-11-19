from __future__ import annotations
import numpy as np

def uniform_indices(n_frames: int, budget: int) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, budget, dtype=int).tolist()

def random_indices(n_frames: int, budget: int, rng: np.random.Generator | None = None) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    rng = rng or np.random.default_rng(42)
    return sorted(rng.choice(n_frames, size=budget, replace=False).tolist())

def adaptive_indices(frames: list[np.ndarray], budget: int,
                     motion_weight: float = 0.5,
                     diversity_weight: float = 0.5) -> list[int]:
    """Greedy frame selection using motion and diversity; inputs are RGB uint8 frames."""
    n = len(frames)
    if n <= 0 or budget <= 0:
        return []
    if n <= budget:
        return list(range(n))
    if motion_weight < 0 or diversity_weight < 0:
        raise ValueError("Weights must be non-negative")
    gray = [np.asarray(f, dtype=np.float32).mean(axis=2) / 255.0 for f in frames]
    motion = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        motion[i] = np.mean(np.abs(gray[i] - gray[i - 1]))
    if motion.max() > 0:
        motion /= motion.max()

    selected = [0]
    while len(selected) < budget:
        best_i, best_score = None, -1.0
        for i in range(1, n):
            if i in selected:
                continue
            diversity = max(float(np.mean(np.abs(gray[i] - gray[j]))) for j in selected)
            score = motion_weight * float(motion[i]) + diversity_weight * diversity
            if score > best_score:
                best_i, best_score = i, score
        selected.append(int(best_i))
    return sorted(selected)
