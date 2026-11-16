
from pathlib import Path
import numpy as np
import torch
from lightglue import LightGlue, SIFT
from lightglue.utils import load_image, rbd

# ========================================
# 1. Configuration
# ========================================
DEVICE = "cuda"
DATA_ROOT = Path("data/hpatches-sequences-release")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLDS = np.arange(1, 11)
TARGET_IMAGES = range(2, 7)

# ========================================
# 2. Load Method B
# ========================================
extractor = SIFT(
    max_num_keypoints=2048
).eval().to(DEVICE)

matcher = LightGlue(
    features="sift"
).eval().to(DEVICE)

# ========================================
# 3. Evaluate one image pair
# ========================================
@torch.inference_mode()
def evaluate_pair(folder, target):
    image0 = load_image(str(folder / "1.ppm")).to(DEVICE)
    image1 = load_image(
        str(folder / f"{target}.ppm")
    ).to(DEVICE)

    feats0 = extractor.extract(image0, resize=1024)
    feats1 = extractor.extract(image1, resize=1024)

    result = matcher({
        "image0": feats0,
        "image1": feats1
    })

    feats0, feats1, result = [
        rbd(x) for x in (feats0, feats1, result)
    ]

    matches = result["matches"]
    n = len(matches)

    if n == 0:
        return np.full(10, np.nan), 0

    points0 = (
        feats0["keypoints"][matches[:, 0]]
        .cpu().numpy()
    )
    points1 = (
        feats1["keypoints"][matches[:, 1]]
        .cpu().numpy()
    )

    H = np.loadtxt(folder / f"H_1_{target}")

    points0_h = np.column_stack([
        points0, np.ones(n)
    ])

    projected_h = (H @ points0_h.T).T
    valid = np.abs(projected_h[:, 2]) > 1e-8

    errors = np.full(n, np.inf)

    projected = (
        projected_h[valid, :2]
        / projected_h[valid, 2:3]
    )

    errors[valid] = np.linalg.norm(
        projected - points1[valid],
        axis=1
    )

    # NumPy array: MMA@1px through MMA@10px
    mma = np.array([
        np.mean(errors <= t)
        for t in THRESHOLDS
    ], dtype=np.float64)

    return mma, n


# ========================================
# 4. Batch evaluation
# ========================================

# Original 20 Pilot sequences
PILOT_SEQUENCES = {
    "i_bologna",
    "i_fog",
    "i_greenhouse",
    "i_kurhaus",
    "i_leuven",
    "i_lionday",
    "i_nescafe",
    "i_nijmegen",
    "i_table",
    "i_whitebuilding",
    "v_bark",
    "v_beyus",
    "v_boat",
    "v_charing",
    "v_colors",
    "v_grace",
    "v_home",
    "v_maskedman",
    "v_underground",
    "v_weapons",
}

# Find all sequences
all_sequences = sorted(
    folder for folder in DATA_ROOT.iterdir()
    if folder.is_dir()
    and folder.name.startswith(("i_", "v_"))
)

# Validate the dataset and split
all_names = {folder.name for folder in all_sequences}

assert len(all_sequences) == 116, (
    f"Expected 116 sequences, found {len(all_sequences)}"
)

assert len(PILOT_SEQUENCES) == 20

assert PILOT_SEQUENCES <= all_names, (
    f"Missing pilot sequences: {PILOT_SEQUENCES - all_names}"
)

# Exclude Pilot sequences from final evaluation
sequences = [
    folder for folder in all_sequences
    if folder.name not in PILOT_SEQUENCES
]

assert len(sequences) == 96

N = len(sequences)

print(f"Pilot sequences excluded: {len(PILOT_SEQUENCES)}")
print(f"Held-out test sequences: {N}")
print(f"Total image pairs: {N * 5}")


mma = np.full((N, 5, 10), np.nan)
num_matches = np.zeros((N, 5), dtype=np.int32)

sequence_names = np.array(
    [folder.name for folder in sequences]
)

sequence_types = np.array([
    "illumination" if folder.name.startswith("i_")
    else "viewpoint"
    for folder in sequences
])

print(f"Found {N} sequences", flush=True)

for i, folder in enumerate(sequences):
    for j, target in enumerate(TARGET_IMAGES):
        scores, count = evaluate_pair(folder, target)

        mma[i, j] = scores
        num_matches[i, j] = count

    print(
        f"[{i+1}/{N}] {folder.name} completed",
        flush=True
    )

# ========================================
# 5. Aggregate results
# ========================================
illum_mask = sequence_types == "illumination"
view_mask = sequence_types == "viewpoint"

def mean_valid(arr, axis=0):
    valid = np.isfinite(arr)
    counts = np.sum(valid, axis=axis)
    totals = np.sum(np.where(valid, arr, 0), axis=axis)
    return np.divide(
        totals, counts,
        out=np.full_like(totals, np.nan, dtype=float),
        where=counts > 0
    )

mma_illumination = mean_valid(mma[illum_mask], axis=0)
mma_viewpoint = mean_valid(mma[view_mask], axis=0)

# Average first over pairs within a sequence,
# then over sequences within each category.
seq_mma = mean_valid(mma, axis=1)

overall_illumination = mean_valid(
    seq_mma[illum_mask], axis=0
)
overall_viewpoint = mean_valid(
    seq_mma[view_mask], axis=0
)

# ========================================
# 6. Save NumPy arrays
# ========================================
np.save(OUTPUT_DIR / "mma.npy", mma)
np.save(OUTPUT_DIR / "num_matches.npy", num_matches)

np.savez(
    OUTPUT_DIR / "method_b_results.npz",
    mma=mma,
    num_matches=num_matches,
    sequence_names=sequence_names,
    sequence_types=sequence_types,
    thresholds=THRESHOLDS,
    mma_illumination=mma_illumination,
    mma_viewpoint=mma_viewpoint,
    overall_illumination=overall_illumination,
    overall_viewpoint=overall_viewpoint
)

# ========================================
# 7. Print NumPy results
# ========================================
np.set_printoptions(precision=4, suppress=True)

print("\n=== Method B Results ===")
print("MMA shape:", mma.shape)
print("Matches shape:", num_matches.shape)

print("\nIllumination MMA@3px by difficulty:")
print(mma_illumination[:, 2])

print("\nViewpoint MMA@3px by difficulty:")
print(mma_viewpoint[:, 2])

print("\nOverall Illumination MMA@3px:")
print(overall_illumination[2])

print("\nOverall Viewpoint MMA@3px:")
print(overall_viewpoint[2])

print("\nResults saved to:", OUTPUT_DIR.resolve())
