from pathlib import Path
import numpy as np
import cv2
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd


# process data
def get_data_root() -> Path:
    """
    Download data with kagglehub, return folder path
    """
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download(
        "javidtheimmortal/hpatches-sequence-release"
    )
    p = Path(path)
    return p / "hpatches-sequences-release"


def list_sequences(root: Path) -> list[str]:
    """
    return catergory name
    """
    return sorted(list(p.name for p in root.iterdir() if p.is_dir()))


def load_pairs(seq_dir: Path, k: int):
    """
    return (img1, imgk, H)
    """
    # k cannot be 1 which is compare to itself
    if k == 1:
        return None

    img1 = cv2.imread(seq_dir / "1.ppm", cv2.IMREAD_GRAYSCALE)
    imgk = cv2.imread(seq_dir / f"{str(k)}.ppm", cv2.IMREAD_GRAYSCALE)
    H = np.loadtxt(seq_dir / f"H_1_{str(k)}")
    return img1, imgk, H


def main():
    p = get_data_root()
    print(f"path: {str(p)}")
    cls_list = list_sequences(p)
    load_pairs(seq_dir=p / f"{cls_list[0]}", k=2)


if __name__ == "__main__":
    main()
