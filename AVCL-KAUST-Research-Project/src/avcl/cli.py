from __future__ import annotations
import argparse
from .data import validate_dataset
from .engine import run_experiment

def _common(p):
    p.add_argument("--video-root", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--split", type=int, default=1, choices=[1,2,3])

def validate_main():
    p = argparse.ArgumentParser(description="Validate UCF101 video paths and official split files")
    _common(p); p.add_argument("--check-files", type=int, default=50)
    a = p.parse_args()
    print(validate_dataset(a.video_root, a.split_dir, a.split, a.check_files))

def run_main():
    p = argparse.ArgumentParser(description="Run class-incremental UCF101 experiments")
    _common(p)
    p.add_argument("--output", default="results/run")
    p.add_argument("--methods", nargs="+", default=["finetune","er","ewc","adaptive"],
                   choices=["finetune","er","ewc","adaptive","adaptive_motion","adaptive_diversity","adaptive_combined"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--frames-per-video", type=int, default=4)
    p.add_argument("--memory-budget-frames", type=int, default=80)
    p.add_argument("--classes-per-task", type=int, default=5)
    p.add_argument("--max-classes", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--ewc-strength", type=float, default=100.0)
    p.add_argument("--image-size", type=int, default=224)
    a = p.parse_args()
    result = run_experiment(a.video_root, a.split_dir, a.output, a.split, tuple(a.methods),
        a.epochs, a.batch_size, a.frames_per_video, a.memory_budget_frames,
        a.classes_per_task, a.max_classes, a.seed, a.pretrained, a.ewc_strength, a.image_size)
    print(result)
