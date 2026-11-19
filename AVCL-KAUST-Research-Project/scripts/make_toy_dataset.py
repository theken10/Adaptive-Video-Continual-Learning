from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create a tiny synthetic UCF101-like dataset for pipeline smoke tests.")
    parser.add_argument("--output", default="data/toy")
    parser.add_argument("--videos-per-class", type=int, default=6)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.output)
    classes = ["MovingCircle", "MovingSquare"]
    split_dir = root.parent / "toy_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "classInd.txt").write_text("1 MovingCircle\n2 MovingSquare\n", encoding="utf-8")
    train_lines, test_lines = [], []
    for class_id, class_name in enumerate(classes, start=1):
        folder = root / class_name
        folder.mkdir(parents=True, exist_ok=True)
        for video_id in range(args.videos_per_class):
            filename = f"v_{class_name}_g{video_id+1:02d}_c01.avi"
            path = folder / filename
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 8, (96, 96))
            if not writer.isOpened():
                raise RuntimeError("OpenCV could not create an AVI writer on this system.")
            for frame_id in range(args.frames):
                frame = np.zeros((96, 96, 3), dtype=np.uint8) + 20
                x = int(12 + frame_id * 70 / max(args.frames - 1, 1))
                y = 48
                if class_id == 1:
                    cv2.circle(frame, (x, y), 10, (30, 180, 240), -1)
                else:
                    cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (220, 90, 40), -1)
                writer.write(frame)
            writer.release()
            rel = f"{class_name}/{filename}"
            if video_id < max(1, args.videos_per_class - 2):
                train_lines.append(f"{rel} {class_id}")
            else:
                test_lines.append(rel)
    (split_dir / "trainlist01.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (split_dir / "testlist01.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    print(f"Toy videos: {root.resolve()}")
    print(f"Toy split files: {split_dir.resolve()}")

if __name__ == "__main__":
    main()
