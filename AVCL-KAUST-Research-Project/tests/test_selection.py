import numpy as np
from avcl.selection import uniform_indices, random_indices, adaptive_indices

def test_uniform_indices():
    assert uniform_indices(10, 4) == [0, 3, 6, 9]
    assert uniform_indices(0, 3) == []

def test_random_indices_are_deterministic():
    assert random_indices(10, 4, np.random.default_rng(7)) == random_indices(10, 4, np.random.default_rng(7))

def test_adaptive_budget_and_order():
    frames = [np.full((8,8,3), i*10, dtype=np.uint8) for i in range(10)]
    idx = adaptive_indices(frames, 4)
    assert len(idx) == 4 and idx == sorted(set(idx))
