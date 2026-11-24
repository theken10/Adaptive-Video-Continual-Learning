
from __future__ import annotations
import numpy as np


def uniform_indices(n_frames: int, budget: int) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    return np.linspace(0, n_frames - 1, budget, dtype=int).tolist()


def random_indices(n_frames: int, budget: int, rng=None) -> list[int]:
    if n_frames <= 0 or budget <= 0:
        return []
    if n_frames <= budget:
        return list(range(n_frames))
    rng = rng if rng is not None else np.random.default_rng(42)
    return sorted(rng.choice(n_frames, size=budget, replace=False).tolist())


def _features(frames):
    gray = [np.asarray(f, dtype=np.float32).mean(axis=2) / 255.0 for f in frames]
    motion = np.zeros(len(gray), dtype=np.float32)
    for i in range(1, len(gray)):
        motion[i] = np.mean(np.abs(gray[i] - gray[i-1]))
    if motion.max() > 0:
        motion /= motion.max()
    desc = np.stack([g.reshape(-1) for g in gray])
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    desc = desc / np.maximum(norms, 1e-8)
    return motion, desc


def _greedy(frames, budget, use_motion=True, use_diversity=True, use_coverage=True):
    n = len(frames)
    if n <= 0 or budget <= 0:
        return []
    if n <= budget:
        return list(range(n))

    motion, desc = _features(frames)
    selected = [0, n-1] if budget >= 2 else [0]

    while len(selected) < budget:
        candidates = [i for i in range(n) if i not in selected]
        scores = []
        for i in candidates:
            diversity = min(float(np.linalg.norm(desc[i] - desc[j])) for j in selected)
            coverage = min(abs(i-j) for j in selected) / max(n-1, 1)
            score = (
                (float(motion[i]) if use_motion else 0.0)
                + (diversity if use_diversity else 0.0)
                + (coverage if use_coverage else 0.0)
            )
            scores.append(score)
        selected.append(candidates[int(np.argmax(scores))])
    return sorted(selected)


def motion_indices(frames, budget):
    """Motion-only frame selection."""
    return _greedy(frames, budget, True, False, False)


def diversity_indices(frames, budget):
    """Visual-diversity-only frame selection."""
    return _greedy(frames, budget, False, True, False)


def adaptive_indices(frames, budget, motion_weight=0.4,
                     diversity_weight=0.4, coverage_weight=0.2):
    """Combined motion, visual diversity, and temporal coverage selection."""
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

    motion, desc = _features(frames)
    selected = [0, n-1] if budget >= 2 else [0]
    while len(selected) < budget:
        candidates = [i for i in range(n) if i not in selected]
        scores = []
        for i in candidates:
            diversity = min(float(np.linalg.norm(desc[i]-desc[j])) for j in selected)
            coverage = min(abs(i-j) for j in selected) / max(n-1, 1)
            scores.append(weights[0]*float(motion[i]) +
                          weights[1]*diversity + weights[2]*coverage)
        selected.append(candidates[int(np.argmax(scores))])
    return sorted(selected)
