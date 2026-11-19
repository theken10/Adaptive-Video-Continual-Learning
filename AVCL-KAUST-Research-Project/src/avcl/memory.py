from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from .selection import uniform_indices, adaptive_indices

@dataclass
class ReplayItem:
    frames: torch.Tensor  # [T,C,H,W], normalized CPU tensor
    label: int
    source: str

class ReplayBuffer:
    """Frame-budgeted class-balanced replay memory."""
    def __init__(self, frame_budget: int, strategy: str = "uniform"):
        if frame_budget < 0:
            raise ValueError("frame_budget must be >= 0")
        if strategy not in ("uniform", "adaptive"):
            raise ValueError("strategy must be uniform or adaptive")
        self.frame_budget = frame_budget
        self.strategy = strategy
        self.items: list[ReplayItem] = []

    @property
    def stored_frames(self) -> int:
        return sum(int(item.frames.shape[0]) for item in self.items)

    def update(self, candidates: list[ReplayItem]):
        if self.frame_budget == 0 or not candidates:
            self.items = []
            return
        by_class: dict[int, list[ReplayItem]] = {}
        for item in self.items + candidates:
            by_class.setdefault(item.label, []).append(item)
        classes = sorted(by_class)
        cursors = {c: 0 for c in classes}
        chosen: list[ReplayItem] = []
        remaining = self.frame_budget
        # Round-robin class-balanced selection; frame budget is strictly enforced.
        while remaining > 0:
            progressed = False
            for cls in classes:
                pool = by_class[cls]
                if cursors[cls] >= len(pool) or remaining <= 0:
                    continue
                item = pool[cursors[cls]]
                cursors[cls] += 1
                n = min(int(item.frames.shape[0]), remaining)
                if n <= 0:
                    continue
                if n < item.frames.shape[0]:
                    if self.strategy == "adaptive":
                        # Convert normalized RGB tensors back to uint8 RGB for the selector.
                        arr = item.frames.detach().cpu().numpy().transpose(0, 2, 3, 1)
                        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                        rgb = [np.clip((frame * std + mean) * 255.0, 0, 255).astype(np.uint8) for frame in arr]
                        indices = adaptive_indices(rgb, n)
                    else:
                        indices = uniform_indices(len(item.frames), n)
                    frames = item.frames[indices].clone()
                else:
                    frames = item.frames.clone()
                chosen.append(ReplayItem(frames, item.label, item.source))
                remaining -= n
                progressed = True
            if not progressed:
                break
        self.items = chosen
