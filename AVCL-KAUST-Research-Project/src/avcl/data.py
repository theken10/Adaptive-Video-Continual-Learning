from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .selection import uniform_indices, random_indices, adaptive_indices

VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv"}

@dataclass(frozen=True)
class VideoRecord:
    path: str
    label: int
    class_name: str

def read_class_index(split_dir: str | Path) -> dict[str, int]:
    path = Path(split_dir) / "classInd.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            mapping[parts[1]] = int(parts[0]) - 1
    if not mapping:
        raise ValueError(f"No class entries found in {path}")
    return mapping

def read_ucf101_split(video_root: str | Path, split_dir: str | Path, split: int = 1,
                      subset: str = "train") -> tuple[list[VideoRecord], dict[str, int]]:
    """Parse official UCF101 recognition split files. Labels returned zero-based."""
    if split not in (1, 2, 3):
        raise ValueError("split must be 1, 2, or 3")
    if subset not in ("train", "test"):
        raise ValueError("subset must be 'train' or 'test'")
    root, sdir = Path(video_root), Path(split_dir)
    class_to_idx = read_class_index(sdir)
    list_name = f"{'trainlist' if subset == 'train' else 'testlist'}{split:02d}.txt"
    list_path = sdir / list_name
    if not list_path.exists():
        raise FileNotFoundError(f"Missing official split file: {list_path}")
    records = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rel = Path(parts[0])
        class_name = rel.parts[0]
        if class_name not in class_to_idx:
            raise ValueError(f"Class {class_name!r} in {list_name} is absent from classInd.txt")
        if subset == "train":
            if len(parts) < 2:
                raise ValueError(f"Training line needs a label: {line}")
            label = int(parts[1]) - 1
            if label != class_to_idx[class_name]:
                raise ValueError(f"Label mismatch for {line!r}; classInd says {class_to_idx[class_name]+1}")
        else:
            label = class_to_idx[class_name]
        full = root / rel
        records.append(VideoRecord(str(full), label, class_name))
    return records, class_to_idx

def decode_video(path: str, max_decode_frames: int = 500) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while cap.isOpened() and len(frames) < max_decode_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Unable to decode video: {path}")
    return frames

class UCF101VideoDataset(Dataset):
    def __init__(self, records: list[VideoRecord], frames_per_video: int = 8,
                 image_size: int = 224, selection: str = "uniform", seed: int = 42):
        self.records = records
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.selection = selection
        self.seed = seed
        if selection not in ("uniform", "random", "adaptive"):
            raise ValueError("selection must be uniform, random, or adaptive")

    def __len__(self):
        return len(self.records)

    def load_record(self, record: VideoRecord, index: int = 0):
        frames = decode_video(record.path)
        if self.selection == "uniform":
            indices = uniform_indices(len(frames), self.frames_per_video)
        elif self.selection == "random":
            indices = random_indices(len(frames), self.frames_per_video,
                                     np.random.default_rng(self.seed + index))
        else:
            indices = adaptive_indices(frames, self.frames_per_video)
        chosen = [frames[i] for i in indices]
        tensors = []
        for frame in chosen:
            image = Image.fromarray(frame).resize((self.image_size, self.image_size))
            arr = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors), record.label, record.path

    def __getitem__(self, index):
        return self.load_record(self.records[index], index)

def collate_videos(batch):
    videos, labels, paths = zip(*batch)
    return list(videos), torch.tensor(labels, dtype=torch.long), list(paths)

def validate_dataset(video_root: str | Path, split_dir: str | Path, split: int = 1,
                     check_files: int = 20) -> dict:
    train, classes = read_ucf101_split(video_root, split_dir, split, "train")
    test, _ = read_ucf101_split(video_root, split_dir, split, "test")
    missing = [r.path for r in (train + test)[:check_files] if not Path(r.path).exists()]
    return {"classes": len(classes), "train_videos": len(train), "test_videos": len(test),
            "checked_paths": min(check_files, len(train) + len(test)), "missing_sample_paths": missing}
