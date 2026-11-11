# scripts: Team experiment code

Store team-written execution, evaluation, aggregation, and plotting scripts here. Official external code lives in ../third_party/.

## Planned functionality

1. Verify pilot/test sequence IDs, counts, categories, and absence of overlap.
2. Load HPatches images and ground-truth homographies, handling resized coordinates consistently.
3. Extract and reuse SIFT features for A/B, then add SuperPoint + LightGlue for C.
4. Compute per-pair MMA@1-10px, match counts, and zero-match statistics.
5. Aggregate and plot results separately for illumination and viewpoint changes.

This is a planned feature list; it does not imply these scripts already exist.

## Coding conventions

- Run from the project root, read parameters from configs/, and read splits from splits/.
- A/B must share identical keypoints and descriptors, including normalization.
- Verify that homographies and keypoints use consistent coordinates before computing errors.
- Save outputs in results/ and record the actual command, configuration, split, and environment.
- Distinguish upstream homography AUC from this project's MMA; they are not interchangeable.

## Check-in relevance

Item 2b requires the exact command or notebook cell that produced the baseline. Item 5 can reference files or commits here as evidence of actual contributions.
