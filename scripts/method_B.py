"""Method B: the same cached SIFT features as A, matched with LightGlue."""
import json
from pathlib import Path
import numpy as np
import torch
from lightglue import LightGlue
from method_A import FeatureStore, extract_keypoints, reprojection_error
from method_C import resolve_device


def build_model(device="cpu", config=None, cache_dir=None):
    root = Path(__file__).resolve().parents[1]
    config = config if config is not None else json.loads((root / "configs/experiment.json").read_text())
    if config["descriptor_normalization"] != "rootsift":
        raise ValueError("Pretrained SIFT LightGlue requires shared RootSIFT descriptors for A and B")
    device = resolve_device(device)
    extractor = FeatureStore(config, cache_dir or root / "data/feature_cache")
    matcher = LightGlue(features="sift", flash=torch.device(device).type == "cuda", mp=False,
                        depth_confidence=-1, width_confidence=-1,
                        filter_threshold=config.get("filter_threshold", 0.1)).eval().to(device)
    return extractor, matcher


def _tensor_features(features, device):
    # FeatureStore records radius; upstream OpenCV SIFT LightGlue expects size.
    arrays = {"keypoints": features.keypoints, "descriptors": features.descriptors,
              "scales": features.scales * 2, "oris": features.orientations,
              "image_size": features.image_size}
    return {key: torch.tensor(value, dtype=torch.float32, device=device)[None]
            for key, value in arrays.items()}


@torch.inference_mode()
def match(extractor, matcher, img1, imgk, device="cpu"):
    device = resolve_device(device)
    f0, f1 = extractor.get(img1), extractor.get(imgk)
    if not len(f0.keypoints) or not len(f1.keypoints):
        return np.empty((0, 2)), np.empty((0, 2))
    out = matcher({"image0": _tensor_features(f0, device), "image1": _tensor_features(f1, device)})
    indices = out["matches"][0].cpu().numpy()
    return f0.keypoints[indices[:, 0]].copy(), f1.keypoints[indices[:, 1]].copy()


if __name__ == "__main__":
    from evaluate import main
    main(default_methods=["B"])
