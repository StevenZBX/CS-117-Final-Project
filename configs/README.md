# configs: Experiment configurations

Store experiment settings here so runs can be reproduced without scattering parameters across code.

## What belongs here

- Evaluation configurations for A/B/C, preferably YAML or JSON.
- Image resizing rules, maximum keypoint count, and A's ratio-test threshold.
- SIFT implementation, descriptor normalization, model weight paths, and random seeds.
- Dataset paths, split file paths, and output paths.
- Error coordinate system, MMA pixel thresholds, zero-match scoring, and averaging rules.
- D's training settings: batch size, learning rate, maximum steps, and GPU-time budget.

## Project conventions

The proposal specifies a maximum image side of 1024 px, up to 2048 keypoints, and a ratio-test threshold of 0.8 for A. These settings still need validation and freezing; they are not measured results. A and B must share the exact same SIFT features, not merely matching parameter values.

Validate and freeze configurations on the pilot split before evaluating the held-out test split. Each result must identify its configuration version. Do not select configurations or checkpoints using test results.

## Check-in relevance

Supports the method description in Item 1 and the reproducible measured baseline in Item 2. Do not store model weights or datasets here.
