"""Method C: SuperPoint + LightGlue, using the shared A/B/C interface."""
import json
from pathlib import Path
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
from dataset_load import errors


def resolve_device(device="auto"):
    """Select CUDA, then MPS, then CPU; explicit unavailable devices fail early."""
    requested = str(device)
    mps = getattr(torch.backends, "mps", None)
    mps_available = mps is not None and mps.is_available()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if mps_available else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {device}; use auto, cpu, cuda, or mps")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use device='cpu' or 'auto'.")
    if selected.type == "mps" and not mps_available:
        raise RuntimeError("MPS unavailable in this PyTorch/macOS environment; use device='cpu' or 'auto'.")
    return str(selected)


def build_model(device="cpu", config=None, cache_dir=None):
    """Return (extractor, matcher). cache_dir is accepted for API compatibility.

    Inputs are prepared by the shared loader; extraction does not resize again.
    C keeps features in memory only and does not use the SIFT disk cache.
    """
    device = resolve_device(device)
    if config is None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/experiment.json").read_text())
    extractor = SuperPoint(max_num_keypoints=config["max_keypoints"]).eval().to(device)
    matcher = LightGlue(features="superpoint", flash=torch.device(device).type == "cuda", mp=False,
                        depth_confidence=-1, width_confidence=-1,
                        filter_threshold=config.get("filter_threshold", 0.1)).eval().to(device)
    return extractor, matcher


@torch.inference_mode()
def extract_keypoints(extractor, image, device="cpu"):
    """Return all detected (N,2) locations in input-image coordinates."""
    device = resolve_device(device)
    features = extractor.extract(numpy_image_to_torch(image).to(device), resize=None)
    return rbd(features)["keypoints"].cpu().numpy().reshape(-1, 2)


@torch.inference_mode()
def match(extractor, matcher, img1, imgk, device="cpu"):
    """Return two matching (N,2) NumPy arrays, including (0,2) when empty."""
    device = resolve_device(device)
    feats1 = extractor.extract(numpy_image_to_torch(img1).to(device), resize=None)
    featsk = extractor.extract(numpy_image_to_torch(imgk).to(device), resize=None)
    out = matcher({"image0": feats1, "image1": featsk})
    feats1, featsk, out = rbd(feats1), rbd(featsk), rbd(out)
    indices = out["matches"]
    return (feats1["keypoints"][indices[:, 0]].cpu().numpy().reshape(-1, 2),
            featsk["keypoints"][indices[:, 1]].cpu().numpy().reshape(-1, 2))


def reprojection_error(kpts0, kpts1, H):
    """Forward pixel errors; points and H must share a coordinate system."""
    return errors(H, kpts0, kpts1)


if __name__ == "__main__":
    from evaluate import main
    main(default_methods=["C"])
