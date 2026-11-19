# Research protocol

## Primary hypothesis
Under a fixed replay-frame budget, motion- and diversity-aware frame selection may preserve useful temporal/visual information better than uniform frame sampling during class-incremental video learning.

## Controlled comparisons
- Use the same official UCF101 split, class order, model initialization, optimizer, epochs, frame count and random seed.
- Compare fine-tuning, ER-uniform, EWC and adaptive ER.
- Run at least 3 seeds for reported comparisons.
- Keep a validation split for tuning; do not tune on the test set.

## Required ablations
1. Uniform selection
2. Random selection
3. Motion-only selection
4. Diversity-only selection
5. Combined motion + diversity

The code currently includes uniform, random selector utility, and combined adaptive selection. Motion-only and diversity-only ablations are explicit future implementation work and must be added before claiming those ablations were conducted.

## Metrics
Average accuracy, forgetting, backward transfer, replay memory frames, wall-clock runtime, and accuracy vs. memory budget.

## Reporting
Report mean ± standard deviation across seeds, hardware, software versions, dataset split, class order, training budget, limitations, and failed experiments. Do not invent or selectively omit results.
