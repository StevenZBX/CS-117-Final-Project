import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
import numpy as np
import csv
import os

from Hpatches_loader import get_data_root, list_sequences, load_pairs


def build_model(device):
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    return extractor, matcher


def match(extractor, matcher, img1, imgk, device):
    """
    extract feature and match them
    return keypoints1 and keypointsn
    """

    max_side = 1024
    resize = max_side if max(img1.shape[:2]) > max_side else None

    # get feature
    feats1 = extractor.extract(
        numpy_image_to_torch(img1).to(device), resize=resize
    )

    featsk = extractor.extract(
        numpy_image_to_torch(imgk).to(device), resize=resize
    )

    # match
    out = matcher({"image0": feats1, "image1": featsk})
    feats1, featsk, out = rbd(feats1), rbd(featsk), rbd(out)
    m = out["matches"]
    kpts1 = feats1["keypoints"][m[:, 0]]
    kptsk = featsk["keypoints"][m[:, 1]]
    kpts1, kptsk = kpts1.cpu().numpy(), kptsk.cpu().numpy()

    return kpts1, kptsk


def reprojection_error(kpts0, kpts1, H) -> np.ndarray:
    ones = np.ones((len(kpts0), 1))
    pts = np.hstack([kpts0, ones])
    proj = pts @ H.T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - kpts1, axis=1)


if __name__ == "__main__":
    device = "cuda"
    p = get_data_root()
    seqs = list_sequences(p)

    err_list = []
    extractor, matcher = build_model(device=device)
    
    rows = []
    for seq in seqs:
        for k in range(2, 7):
            img1, imgk, H = load_pairs(p / seq, k)
            kpts1, kptsk = match(extractor, matcher, img1, imgk, device)
            error = reprojection_error(kpts1, kptsk, H)

            row = {
                "sequence": seq,
                "category": "illumination" if seq.startswith("i_") else "viewpoint",
                "k": k,
                "num_matches": len(error),
            }
            for t in range(1, 11):
                row[f"mma@{t}"] = float((error <= t).mean()) if len(error) else 0.0
            rows.append(row)

    os.makedirs("results/C_superpoint_lightglue", exist_ok=True)
    with open("results/C_superpoint_lightglue/per_pair.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    
