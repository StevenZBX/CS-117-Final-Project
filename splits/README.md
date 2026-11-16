# splits: HPatches sequence manifests

- `hpatches_pilot.txt`: 20 sequences, 10 illumination + 10 viewpoint; 120 unique images and 100 pairs.
- `hpatches_test.txt`: remaining 96 sequences, 47 illumination + 49 viewpoint; 576 unique images and 480 pairs.

Each sequence contributes image 1 paired with images 2-6. The lists were reconstructed from the original notebook's sorted sequence list and Python random seed 0, sampling ten illumination sequences followed by ten viewpoint sequences. The original per-pair CSV was unavailable for independent verification of those IDs.

Run `python scripts/evaluate.py --prepare-splits` to validate/recover these lists from the dataset. Different existing lists are never replaced. Counts are printed to the console; no JSON statistics file is generated. All evaluations use this same separation.
