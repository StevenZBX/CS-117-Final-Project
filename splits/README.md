# HPatches train/test manifests

Train: 96 sequences (47 illumination, 49 viewpoint), 576 images, 480 pairs.
Test: 20 sequences (10 illumination, 10 viewpoint), 120 images, 100 pairs.

All original 20 pilot sequences belong to train. Test is sampled from the former test pool, using Python random.Random(0), first ten illumination then ten viewpoint sequences from sorted lists. Train is the complement. The manifests are disjoint and cover all 116 sequences. Each sequence contributes pairs 1→2 through 1→6. Original pilot IDs are recorded in dataset_load.ORIGINAL_PILOT for membership validation.

Previous 96-sequence test results do not represent this split. Some new test sequences were evaluated previously; this split is not a claim of unseen data.
