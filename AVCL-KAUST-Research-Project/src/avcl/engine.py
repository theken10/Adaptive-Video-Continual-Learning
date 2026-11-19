from __future__ import annotations
from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from .data import UCF101VideoDataset, read_ucf101_split, collate_videos
from .models import VideoClassifier
from .memory import ReplayBuffer, ReplayItem
from .regularizers import EWC
from .metrics import average_accuracy, forgetting, backward_transfer

def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MemoryDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        return item.frames, item.label, item.source

def _loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_videos)

def train_epoch(model, loader, optimizer, device, ewc=None, allowed_classes=None):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, total, correct = 0.0, 0, 0
    for videos, labels, _ in loader:
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model([v.to(device) for v in videos], allowed_classes)
        loss = criterion(logits, labels)
        if ewc is not None:
            loss = loss + ewc.penalty(model)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, dataset, device, batch_size, allowed_classes=None):
    loader = _loader(dataset, batch_size, False)
    model.eval()
    correct = total = 0
    for videos, labels, _ in loader:
        labels = labels.to(device)
        logits = model([v.to(device) for v in videos], allowed_classes)
        correct += int((logits.argmax(1) == labels).sum())
        total += len(labels)
    return correct / total if total else float("nan")

def build_tasks(records, class_names, classes_per_task):
    tasks = []
    for start in range(0, len(class_names), classes_per_task):
        names = class_names[start:start + classes_per_task]
        ids = {name for name in names}
        tasks.append((names, [r for r in records if r.class_name in ids]))
    return tasks

def run_experiment(video_root, split_dir, output, split=1, methods=("finetune","er","ewc","adaptive"),
                   epochs=2, batch_size=4, frames_per_video=4, memory_budget_frames=80,
                   classes_per_task=5, max_classes=None, seed=42, pretrained=False,
                   ewc_strength=100.0, image_size=224):
    seed_everything(seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_records, class_map = read_ucf101_split(video_root, split_dir, split, "train")
    test_records, _ = read_ucf101_split(video_root, split_dir, split, "test")
    class_names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]
    if max_classes:
        class_names = class_names[:max_classes]
    selected = set(class_names)
    train_records = [r for r in train_records if r.class_name in selected]
    test_records = [r for r in test_records if r.class_name in selected]
    tasks = build_tasks(train_records, class_names, classes_per_task)
    test_tasks = build_tasks(test_records, class_names, classes_per_task)
    if not tasks or any(not records for _, records in tasks):
        raise ValueError("At least one selected task has no training videos. Check dataset paths/split.")
    num_classes = len(class_map)
    rows = []

    for method in methods:
        seed_everything(seed)
        model = VideoClassifier(num_classes, pretrained=pretrained).to(device)
        buffer = ReplayBuffer(memory_budget_frames, "adaptive" if method == "adaptive" else "uniform")
        ewc = None
        acc = np.full((len(tasks), len(tasks)), np.nan, dtype=float)
        started = time.time()

        for task_id, (task_names, records) in enumerate(tasks):
            seen_count = min((task_id + 1) * classes_per_task, len(class_names))
            seen_class_ids = [class_map[name] for name in class_names[:seen_count]]
            selection = "adaptive" if method == "adaptive" else "uniform"
            train_ds = UCF101VideoDataset(records, frames_per_video, image_size, selection, seed)
            current_loader = _loader(train_ds, batch_size, True)
            datasets = [train_ds]
            if method in ("er", "adaptive") and buffer.items:
                datasets.append(MemoryDataset(buffer.items))
            combined = ConcatDataset(datasets) if len(datasets) > 1 else train_ds
            train_loader = _loader(combined, batch_size, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            for _ in range(epochs):
                train_epoch(model, train_loader, optimizer, device, ewc if method == "ewc" else None, seen_class_ids)

            if method == "ewc":
                if ewc is None:
                    ewc = EWC(model, ewc_strength)
                ewc.estimate(model, current_loader, device, seen_class_ids)

            for eval_id in range(task_id + 1):
                eval_names, eval_records = test_tasks[eval_id]
                eval_ds = UCF101VideoDataset(eval_records, frames_per_video, image_size, "uniform", seed)
                acc[task_id, eval_id] = evaluate(model, eval_ds, device, batch_size, seen_class_ids)

            if method in ("er", "adaptive"):
                candidates = []
                for i in range(len(train_ds)):
                    frames, label, path = train_ds[i]
                    candidates.append(ReplayItem(frames.detach().cpu(), int(label), path))
                buffer.update(candidates)

            rows.append({
                "method": method, "task": task_id, "classes_seen": len(class_names[:(task_id+1)*classes_per_task]),
                "average_accuracy": average_accuracy(acc, task_id),
                "forgetting": forgetting(acc, task_id),
                "backward_transfer": backward_transfer(acc, task_id),
                "stored_frames": buffer.stored_frames,
                "elapsed_seconds": time.time() - started
            })

        np.savetxt(output / f"accuracy_matrix_{method}.csv", acc, delimiter=",", fmt="%.6f")
        np.save(output / f"accuracy_matrix_{method}.npy", acc)

    df = pd.DataFrame(rows)
    df.to_csv(output / "metrics.csv", index=False)
    summary = {
        "device": str(device), "split": split, "classes": class_names,
        "methods": list(methods), "epochs": epochs, "batch_size": batch_size,
        "frames_per_video": frames_per_video, "memory_budget_frames": memory_budget_frames,
        "classes_per_task": classes_per_task, "seed": seed, "pretrained": pretrained,
        "note": "Results are generated by this run; do not treat them as validated research claims without repeated controlled experiments.",
        "final_metrics": df.groupby("method").tail(1).replace({np.nan: None}).to_dict(orient="records")
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    if not df.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            part = df[df.method == method]
            ax.plot(part["task"], part["average_accuracy"], marker="o", label=method)
        ax.set_xlabel("Training task"); ax.set_ylabel("Average accuracy")
        ax.set_title("Class-incremental video learning")
        ax.set_ylim(0, 1); ax.grid(True, alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(output / "accuracy_by_task.png", dpi=160); plt.close(fig)
    return summary
