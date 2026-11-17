"""Method A and shared A/B SIFT extraction, normalization, and feature cache."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from dataset_load import errors


@dataclass(frozen=True)
class Features:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scales: np.ndarray  # OpenCV keypoint radius (size / 2)
    orientations: np.ndarray  # radians
    image_size: np.ndarray  # width, height

    def __post_init__(self):
        n = len(self.keypoints)
        if self.keypoints.shape != (n, 2) or self.descriptors.shape != (n, 128):
            raise ValueError("Invalid SIFT feature dimensions")
        if self.scales.shape != (n,) or self.orientations.shape != (n,):
            raise ValueError("Invalid SIFT metadata dimensions")
        for value in vars(self).values():
            value.setflags(write=False)


def extract_sift(image, config):
    sift = cv2.SIFT_create(nfeatures=config["max_keypoints"], **config["sift"])
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        keypoints, descriptors = [], np.empty((0, 128), np.float32)
    order = sorted(range(len(keypoints)), key=lambda i: (-keypoints[i].response, i))[:config["max_keypoints"]]
    selected = [keypoints[i] for i in order]
    desc = descriptors[order].copy()
    if config["descriptor_normalization"] == "rootsift":
        desc = np.sqrt(np.maximum(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-6), 1e-6))
        desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-6)
    elif config["descriptor_normalization"] != "raw":
        raise ValueError("descriptor_normalization must be raw or rootsift")
    return Features(np.array([k.pt for k in selected], dtype=np.float64).reshape(-1, 2),
                    desc, np.array([k.size / 2 for k in selected], np.float32),
                    np.deg2rad(np.array([k.angle for k in selected], np.float32)),
                    np.array([image.shape[1], image.shape[0]], np.int64))


class FeatureStore:
    def __init__(self, config, cache_dir):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory = {}
        extraction = {k: config[k] for k in ("max_keypoints", "sift", "descriptor_normalization")}
        self.fingerprint = json.dumps({"config": extraction, "opencv": cv2.__version__,
            "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, sort_keys=True).encode()

    def get(self, image):
        digest = hashlib.sha256(self.fingerprint + str(image.shape).encode() + image.tobytes()).hexdigest()
        if digest not in self.memory:
            path = self.cache_dir / f"sift_{digest}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as data:
                    value = Features(**{name: data[name] for name in Features.__dataclass_fields__})
            else:
                value = extract_sift(image, self.config)
                temp = path.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **vars(value))
                temp.replace(path)
            self.memory[digest] = value
        return self.memory[digest]

    def clear_memory(self):
        self.memory.clear()


class RatioMatcher:
    name = "A"
    label = "SIFT + ratio test"

    def __init__(self, config):
        self.ratio = config["ratio"]

    def match(self, features0, features1):
        if len(features0.descriptors) == 0 or len(features1.descriptors) < 2:
            return np.empty((0, 2), dtype=np.int64)
        neighbors = cv2.BFMatcher(cv2.NORM_L2).knnMatch(features0.descriptors, features1.descriptors, k=2)
        pairs = [(m.queryIdx, m.trainIdx) for pair in neighbors if len(pair) == 2
                 for m, n in [pair] if m.distance < self.ratio * n.distance]
        return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def build_model(device="cpu", config=None, cache_dir=None):
    """SIFT always uses CPU; device is accepted for the common method API."""
    root = Path(__file__).resolve().parents[1]
    if config is None:
        config = json.loads((root / "configs/experiment.json").read_text())
    return FeatureStore(config, cache_dir or root / "data/feature_cache"), RatioMatcher(config)


def extract_keypoints(extractor, image, device="cpu"):
    return extractor.get(image).keypoints.copy()


def match(extractor, matcher, img1, imgk, device="cpu"):
    f0, f1 = extractor.get(img1), extractor.get(imgk)
    indices = matcher.match(f0, f1)
    return f0.keypoints[indices[:, 0]].copy(), f1.keypoints[indices[:, 1]].copy()


def reprojection_error(kpts0, kpts1, H):
    return errors(H, kpts0, kpts1)

if __name__ == "__main__":
    from evaluate import main
    main(default_methods=["A"])