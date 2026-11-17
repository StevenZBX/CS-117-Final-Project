"""HPatches acquisition, fixed splits, resized pairs, and geometry helpers."""
import random
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]



ORIGINAL_PILOT = ('i_bologna', 'i_fog', 'i_greenhouse', 'i_kurhaus', 'i_leuven', 'i_lionday', 'i_nescafe', 'i_nijmegen', 'i_table', 'i_whitebuilding', 'v_bark', 'v_beyus', 'v_boat', 'v_charing', 'v_colors', 'v_grace', 'v_home', 'v_maskedman', 'v_underground', 'v_weapons')


def split_sequences(names):
    """Deterministic 96/20 split; all original pilot sequences remain in train."""
    names = sorted(names)
    if len(set(names)) != 116 or not set(ORIGINAL_PILOT) <= set(names):
        raise ValueError("Expected all 116 HPatches sequences, including the original pilot")
    eligible = sorted(set(names) - set(ORIGINAL_PILOT))
    rng = random.Random(0)
    test = sorted(rng.sample([x for x in eligible if x.startswith("i_")], 10)
                  + rng.sample([x for x in eligible if x.startswith("v_")], 10))
    return {"train": sorted(set(names) - set(test)), "test": test}


def prepare_splits(data):
    for split, ids in split_sequences(list_sequences(data)).items():
        path = ROOT / "splits" / f"hpatches_{split}.txt"
        if path.exists() and path.read_text().splitlines() != ids:
            raise ValueError(f"Refusing to replace different manifest: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(ids) + "\n")
        validate_splits(data, split)
        print(f"{split}: {len(ids)} sequences, {len(ids)*5} pairs")

def load_image(path, max_side):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    h, w = image.shape
    scale = min(1., max_side / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    nh, nw = image.shape
    return image, np.diag([nw/w, nh/h, 1.])

def errors(H, p0, p1):
    if not len(p0):
        return np.empty(0)
    q = np.column_stack([p0, np.ones(len(p0))]) @ H.T
    valid = np.isfinite(q).all(axis=1) & (np.abs(q[:,2]) > 1e-12)
    result = np.full(len(p0), np.inf)
    result[valid] = np.linalg.norm(q[valid,:2]/q[valid,2,None] - p1[valid], axis=1)
    return result

def score(err, threshold):
    return float(np.mean(err < threshold)) if len(err) else 0.




def validate_splits(data, split):
    if split not in {"train", "test"}:
        raise ValueError("Split must be train or test")
    manifests = {name: ROOT / "splits" / f"hpatches_{name}.txt" for name in ("train", "test")}
    ids = {name: path.read_text().splitlines() for name, path in manifests.items()}
    for name, count, illumination in (("train", 96, 47), ("test", 20, 10)):
        if len(ids[name]) != count or len(set(ids[name])) != count:
            raise ValueError(f"Invalid {name} manifest")
        if sum(x.startswith("i_") for x in ids[name]) != illumination or sum(x.startswith("v_") for x in ids[name]) != count-illumination:
            raise ValueError(f"Invalid illumination/viewpoint counts in {name}")
    if set(ids["train"]) & set(ids["test"]):
        raise ValueError("Train/test overlap")
    if not set(ORIGINAL_PILOT) <= set(ids["train"]):
        raise ValueError("All original pilot sequences must remain in train")
    actual = set(list_sequences(data))
    if set(ids["train"]) | set(ids["test"]) != actual:
        raise ValueError("Manifests do not partition the dataset")
    if ids != split_sequences(actual):
        raise ValueError("Manifests differ from the fixed seeded split")
    for sequence in ids[split]:
        for name in [f"{k}.ppm" for k in range(1, 7)] + [f"H_1_{k}" for k in range(2, 7)]:
            if not (Path(data) / sequence / name).is_file():
                raise FileNotFoundError(Path(data) / sequence / name)
    return ids[split], manifests[split]


def get_data_root(data=None):
    """Use a supplied/local dataset, otherwise obtain HPatches via kagglehub."""
    if data is not None:
        root = Path(data).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        return root
    local = ROOT / "data/hpatches-sequences-release"
    if local.is_dir():
        return local
    import kagglehub
    downloaded = Path(kagglehub.dataset_download("javidtheimmortal/hpatches-sequence-release"))
    candidates = [downloaded, downloaded / "hpatches-sequences-release"]
    candidates += sorted(downloaded.rglob("hpatches-sequences-release"))
    for candidate in candidates:
        if candidate.is_dir() and len(list_sequences(candidate)) == 116:
            return candidate
    raise FileNotFoundError(f"Cannot locate the 116 HPatches sequences under {downloaded}")


def list_sequences(root):
    return sorted(p.name for p in Path(root).iterdir()
                  if p.is_dir() and p.name.startswith(("i_", "v_")))


def load_pairs(seq_dir, k, max_side=1024):
    """Return resized images and their consistently transformed homography."""
    if k not in range(2, 7):
        raise ValueError("Target index must be 2 through 6")
    seq_dir = Path(seq_dir)
    image0, S0 = load_image(seq_dir / "1.ppm", max_side)
    image1, S1 = load_image(seq_dir / f"{k}.ppm", max_side)
    H = np.loadtxt(seq_dir / f"H_1_{k}")
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError(f"Invalid homography: {seq_dir.name}/{k}")
    return image0, image1, S1 @ H @ np.linalg.inv(S0)
