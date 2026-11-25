# Adaptive Memory-Efficient Video Continual Learning

**Research prototype | Computer Vision · Continual Learning · Memory-Efficient Learning**

This repository studies class-incremental video action recognition under a limited replay-memory budget. It compares sequential fine-tuning, Experience Replay (ER), Elastic Weight Consolidation (EWC), and an adaptive replay strategy that selects frames using motion and visual diversity.

> **Research integrity:** This is an implementation framework, not evidence that the proposed adaptive strategy is superior. Run controlled experiments, retain raw outputs, and report only measured results. A completed codebase does not guarantee an internship or publication.

## Research question

Under the same replay-frame budget, does selecting frames according to motion and visual diversity improve the accuracy–memory trade-off and reduce forgetting compared with uniform selection?

## What is implemented

- Official UCF101 recognition split parser (`classInd`, `trainlistXX`, `testlistXX`)
- Video decoding and uniform / random / adaptive frame selection
- ResNet-18 frame encoder with mean-pooled video representations
- Class-incremental task sequence (configurable classes per task)
- Sequential fine-tuning baseline
- Experience Replay with a frame-count memory budget
- Diagonal Fisher Elastic Weight Consolidation
- Adaptive replay memory using motion + visual diversity
- Average accuracy, forgetting, backward transfer, memory frames and runtime
- CSV metrics, JSON experiment configuration and plots
- Toy-video generator and unit tests
- Colab quick-start instructions

## 1. Environment

Python 3.10 or 3.11 is recommended. A CUDA GPU is helpful but not mandatory.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If Windows Application Control blocks native DLLs (for example OpenCV or Pandas), do not bypass your organization's security policy. Use the provided Google Colab workflow instead.

## 2. Dataset layout

Download UCF101 from the [official UCF101 page](https://www.crcv.ucf.edu/data/UCF101.php) and the **Train/Test Splits for Action Recognition** archive from the same page.

After extraction, arrange:

```text
data/
  UCF101/
    ApplyEyeMakeup/
      v_ApplyEyeMakeup_g01_c01.avi
      ...
    ApplyLipstick/
      ...
  ucfTrainTestlist/
    classInd.txt
    trainlist01.txt
    testlist01.txt
    trainlist02.txt
    testlist02.txt
    ...
```

The dataset files are not included in this repository. UCF101 is large and has its own terms; download it from the official source. Do not commit the dataset to GitHub.

Validate paths and split files:

```bash
avcl-validate --video-root data/UCF101 --split-dir data/ucfTrainTestlist --split 1
```

## 3. Pipeline smoke test (no UCF101 download required)

Generate a tiny synthetic two-class video dataset:

```bash
python scripts/make_toy_dataset.py --output data/toy --videos-per-class 6
```

Then run one short pass through the complete training/evaluation pipeline:

```bash
avcl-run --video-root data/toy --split-dir data/toy_splits --output results/toy \
  --split 1 --classes-per-task 1 --epochs 1 --frames-per-video 3 \
  --memory-budget-frames 8 --methods finetune er ewc adaptive
```

This confirms that the code path runs. Synthetic data is only a software smoke test and is not research evidence.

## 4. Run a small experiment first

Use a small subset to verify the pipeline before running all 101 classes:

```bash
avcl-run \
  --video-root data/UCF101 \
  --split-dir data/ucfTrainTestlist \
  --output results/ucf101_small \
  --split 1 \
  --max-classes 5 \
  --classes-per-task 2 \
  --epochs 1 \
  --frames-per-video 4 \
  --memory-budget-frames 40 \
  --methods finetune er ewc adaptive
```

Windows PowerShell users can put the command on one line, or use `python scripts/run_experiment.py ...`.

For a more meaningful experiment, increase the number of classes, epochs, seeds, and memory budget only after the small run succeeds. Report mean and standard deviation across at least three seeds for serious comparisons.

## 5. Google Colab (recommended for GPU)

1. Upload this ZIP to Colab or clone the GitHub repository.
2. Mount Google Drive and upload/extract UCF101 into Drive (do not upload the 6.5 GB dataset to GitHub).
3. In a Colab cell:

```python
!git clone YOUR_GITHUB_REPOSITORY_URL
%cd AVCL-KAUST-Research-Project
!pip install -r requirements.txt
!pip install -e .
```

4. Mount Drive and set paths:

```python
from google.colab import drive
drive.mount('/content/drive')

VIDEO_ROOT = '/content/drive/MyDrive/AVCL_DATA/UCF101'
SPLIT_DIR = '/content/drive/MyDrive/AVCL_DATA/ucfTrainTestlist'
```

5. Start with a small run:

```python
!avcl-validate --video-root "$VIDEO_ROOT" --split-dir "$SPLIT_DIR" --split 1
!avcl-run --video-root "$VIDEO_ROOT" --split-dir "$SPLIT_DIR" \
  --output /content/drive/MyDrive/AVCL_RESULTS/small \
  --split 1 --max-classes 5 --classes-per-task 2 --epochs 1 \
  --frames-per-video 4 --memory-budget-frames 40 \
  --methods finetune er ewc adaptive
```

Use Colab's Runtime → Change runtime type → T4 GPU when available. GPU availability depends on the account and current resource limits.

## 6. Methods and fair comparisons

| Method | Continual-learning mechanism | Replay selection |
|---|---|---|
| `finetune` | Sequential training only | None |
| `er` | Experience Replay | Uniform |
| `ewc` | Diagonal Fisher regularization | None |
| `adaptive` | Experience Replay | Motion + visual diversity |

All replay methods use the same total frame budget. For a fair study, keep dataset split, class order, model initialization, optimizer, training epochs, input resolution and random seed fixed.

The adaptive selector is a transparent heuristic, not a claimed novel algorithm. Conduct a literature review before making novelty claims. Recommended ablations:
- motion-only selection
- diversity-only selection
- combined motion + diversity
- uniform selection
- random selection

## 7. Outputs

Each experiment writes:
- `metrics.csv` — per-task accuracy, average accuracy, forgetting, backward transfer, memory frames and elapsed seconds
- `summary.json` — run configuration and final metrics
- `accuracy_by_task.png` — average accuracy by task
- `accuracy_matrix_<method>.csv` — task-by-task accuracy matrix

## 8. Research checklist

- [ ] Verify UCF101 split files and dataset paths
- [ ] Run toy-data smoke test
- [ ] Run a small UCF101 subset
- [ ] Fix and publish class order and split configuration
- [ ] Run at least 3 seeds
- [ ] Add random and ablation baselines
- [ ] Report mean ± standard deviation
- [ ] Discuss limitations, compute budget and failure cases
- [ ] Read and cite the original continual-learning and video-learning papers

## 9. Starting references

- Soomro, Zamir & Shah, **UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild**, CRCV-TR-12-01 (2012). [Official dataset page](https://www.crcv.ucf.edu/data/UCF101.php)
- Villa et al., **vCLIMB: A Novel Video Class Incremental Learning Benchmark**, CVPR 2022. [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2022/papers/Villa_vCLIMB_A_Novel_Video_Class_Incremental_Learning_Benchmark_CVPR_2022_paper.pdf)
- Kirkpatrick et al., **Overcoming catastrophic forgetting in neural networks**, PNAS (2017). [DOI](https://doi.org/10.1073/pnas.1611835114)
- Rebuffi et al., **iCaRL: Incremental Classifier and Representation Learning**, CVPR 2017. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html)

Check paper licenses, dataset terms, and benchmark protocols before publication.


## Current Research Status

- Dataset loading filters missing video files and retains classes available in both splits.
- Adaptive frame selection combines motion, visual diversity, and temporal coverage.
- Unit tests: 5 tests passed in the current development run.

### Dataset and pilot limitations
The local UCF101 copy is incomplete. The available Split 1 data was previously
checked as 56 usable classes. The recorded pilot used 10 classes (842 training
videos and 346 test videos), one epoch, and one seed. These are preliminary
results, not validated benchmark conclusions.

| Method | Accuracy | Forgetting | Stored frames |
|---|---:|---:|---:|
| Fine-tuning | 23.67% | 32.20% | 0 |
| Experience Replay | 18.34% | 34.46% | 200 |
| EWC | 20.12% | 32.20% | 0 |
| Adaptive V1 | 17.16% | 35.03% | 200 |

Adaptive V2 was separately run on CPU: 17.75% accuracy and 33.90% forgetting.
Because its device differed from the earlier GPU runs, it is not a controlled
comparison.

### Remaining research
Repeat experiments with multiple seeds and controlled hardware, evaluate more
classes, perform component ablations, and report mean and standard deviation.


## UCF101 Preliminary Pilot Results

A preliminary single-seed pilot was run on 10 UCF101 classes (Split 1), using 5 tasks, 2 classes per task, and 1 epoch.

| Method | Accuracy | Forgetting | Stored Frames |
|---|---:|---:|---:|
| Fine-tuning | 12.86% | 77.01% | 0 |
| ER | 12.05% | 58.62% | 200 |
| EWC | 12.86% | 76.06% | 0 |
| Adaptive Combined | 17.82% | 52.97% | 200 |

These are preliminary results only, not a validated benchmark. Only one seed and one epoch were used. Repeated-seed experiments and ablations remain necessary. The aggregate metrics were recovered from notebook output; original task-wise matrices were not backed up.

See `results/kaust_pilot/RESEARCH_REPORT.md` for methodology, limitations, and next steps.
