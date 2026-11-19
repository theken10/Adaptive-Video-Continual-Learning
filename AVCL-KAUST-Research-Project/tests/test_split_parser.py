from pathlib import Path
from avcl.data import read_ucf101_split

def test_official_split_parser(tmp_path):
    video_root = tmp_path / "videos"
    split_dir = tmp_path / "splits"
    (video_root / "ClassA").mkdir(parents=True)
    split_dir.mkdir()
    (split_dir / "classInd.txt").write_text("1 ClassA\n")
    (split_dir / "trainlist01.txt").write_text("ClassA/v_ClassA_g01_c01.avi 1\n")
    (split_dir / "testlist01.txt").write_text("ClassA/v_ClassA_g02_c01.avi\n")
    train, mapping = read_ucf101_split(video_root, split_dir, 1, "train")
    test, _ = read_ucf101_split(video_root, split_dir, 1, "test")
    assert mapping == {"ClassA": 0}
    assert train[0].label == test[0].label == 0
