# Adaptive Memory-Efficient Video Continual Learning

## Abstract
This project investigates memory-efficient video continual learning under a class-incremental setting. It evaluates a proposed adaptive frame-selection approach combining motion, diversity, and temporal coverage against fine-tuning, Experience Replay (ER), and Elastic Weight Consolidation (EWC).

## Research Question
Can adaptive selection of representative video frames improve the accuracy-retention trade-off under a fixed replay-memory budget?

## Dataset and Experimental Setup
- Dataset: UCF101
- Official split: Split 1
- Selected classes: 10
- Classes: HorseRace, HulaHoop, JavelinThrow, JugglingBalls, JumpingJack, JumpRope, Kayaking, LongJump, Mixing, MoppingFloor
- Available videos: 842 training, 346 testing
- Tasks: 5 tasks, 2 classes per task
- Epochs: 1
- Random seed: 42
- Device: NVIDIA Tesla T4
- Replay memory budget: 200 frames
- Methods: Fine-tuning, ER, EWC, Adaptive Combined

## Preliminary Results

| Method | Accuracy | Forgetting | Backward Transfer | Stored Frames | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| Fine-tuning | 12.86% | 77.01% | -0.7701 | 0 | 339.04 |
| ER | 12.05% | 58.62% | -0.5862 | 200 | 176.95 |
| EWC | 12.86% | 76.06% | -0.7606 | 0 | 186.91 |
| Adaptive Combined | 17.82% | 52.97% | -0.5297 | 200 | 952.04 |

## Observations
In this single pilot run, Adaptive Combined achieved the highest reported average accuracy (17.82%) and lowest reported forgetting (52.97%) among the four methods. Compared with fine-tuning, its accuracy was approximately 4.96 percentage points higher and forgetting approximately 24.04 percentage points lower.

The adaptive method required approximately 952 seconds, making runtime an important trade-off for further optimization.

## Limitations
1. This is a preliminary experiment using only 10 of the 101 UCF101 classes.
2. Only one random seed and one training epoch were used.
3. Results have not been validated across repeated runs.
4. The original experiment accuracy matrices were not preserved in the Drive backup. The reported aggregate metrics were recovered from the Colab execution output and saved separately.
5. The ablation smoke tests are software checks, not scientific ablation results.
6. The results do not establish statistical significance or generalization to the full UCF101 dataset.

## Next Research Steps
1. Repeat experiments with at least three random seeds.
2. Run motion-only, diversity-only, and combined adaptive methods under identical settings.
3. Preserve all task-wise accuracy matrices and raw logs.
4. Evaluate additional class subsets and training epochs.
5. Investigate the adaptive method's runtime overhead.
6. Report mean and standard deviation across seeds.

## Reproducibility Note
The recovered aggregate metrics are stored in `results/kaust_pilot/recovered_pilot_metrics.csv` and `recovered_pilot_summary.json`. These are reconstructed records from the notebook output, not the original experiment-generated files.
