# CS117 Final Project

A: SIFT + ratio test; B: SIFT + pretrained LightGlue; C: SuperPoint + pretrained LightGlue; D (optional): fine-tuned SIFT + LightGlue.

## Dependencies and installation

Run commands from this project directory. Use Python 3.10 or 3.11 (the training dependency scikit-learn~=1.3.0 is old). Keep evaluation and training in separate virtual environments.

### A-C evaluation

```bash
python3.11 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
python -c "import cv2, torch, lightglue; print(torch.__version__); print(cv2.SIFT_create())"
```

### D training (optional)

In a new terminal, from the project root:

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-train.txt
python -m pip install --no-deps -e ./third_party/glue-factory
python -m pip check
python -c "import torch, lightglue, gluefactory, pycolmap; print(torch.__version__)"
```

The training requirements include every upstream runtime dependency, replacing the floating GitHub LightGlue dependency with our local checkout. The final --no-deps step avoids fetching another copy of LightGlue. pycolmap supports the official SIFT training configuration; unrelated line-feature and pose-estimation extras are excluded. Some transitive dependencies also install OpenCV headless; both OpenCV distributions are constrained below 4.12 in the training environment to retain NumPy 1.x compatibility.

These files are dependency specifications, not a fully tested lockfile. No environment has been installed or inference/training executed during repository setup. Validate the commands on the actual target machine before experiments; API compatibility, especially pycolmap, still needs a smoke test. On a CUDA machine, install the appropriate matching PyTorch/torchvision build for that machine before these requirements. macOS does not provide CUDA; use the planned Colab/NVIDIA environment for GPU training.

After a successful smoke test, save `python -m pip freeze` together with the Python version, hardware, and run command. Do not share a frozen environment blindly between macOS and Colab.

## External sources

Official repositories and exact cloned commits are recorded in `third_party_versions.json`. Third-party code is ignored by the team repository. To reproduce the checkout, clone each URL into its recorded path, then check out its recorded commit. Preserve the upstream license files.

Datasets and pretrained weights are not included or downloaded by this setup. Record their source and version when added.

## Experiment notes

- Recover the original 20 pilot sequence IDs; reserve the other 96 HPatches sequences for testing.
- A and B must share identical SIFT keypoints and descriptors, including any descriptor normalization.
- Freeze resize/homography handling, pixel-threshold coordinates, and zero-match scoring before test evaluation.
- The upstream Glue Factory HPatches evaluation reports homography metrics; implement/verify the project's MMA metric rather than treating that output as MMA.
- Save measured outputs in `results/` and document their exact command.
