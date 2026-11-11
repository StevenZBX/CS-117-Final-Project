# splits: Dataset split manifests

Store dataset split IDs and statistics here, not image files.

## HPatches

- Pilot: the original 20 sequences already used for development and validation.
- Test: the remaining 96 sequences, reserved as the held-out test split.
- Each sequence provides five pairs: image 1 matched with images 2-6.
- Pilot contains 120 unique images and 100 pairs; test contains 576 unique images and 480 pairs.

Recover the actual IDs of the original 20 pilot sequences. Do not replace them with a newly sampled set. The two splits must not overlap. Count illumination/viewpoint sequences separately within each split; the full dataset's 57/59 counts are not the test split counts. The fallback must use the same separation.

## Suggested files (not yet created)

- hpatches_pilot.txt: one actual pilot sequence ID per line.
- hpatches_test.txt: one remaining sequence ID per line.
- split_stats.json: sequence, category, image, and pair counts for each split.
- If pursuing D, separate COCO train/validation source-image ID lists to prevent source-image overlap.

## Check-in relevance

Item 1c explains how the proposal feedback on data splitting was addressed. Item 2a must identify the baseline split and image count accurately. Do not report unverified IDs or statistics as experimental facts.
