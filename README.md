# Project overview

This project compares SIFT + ratio test (A), SIFT + LightGlue (B), and SuperPoint + LightGlue (C) on HPatches. It selects each method's best matching parameters on 96 train sequences and evaluates them on 20 separate test sequences, with CSV results and notebook visualizations.

# Environment setup

Use Python 3.10+ and Git. Open a terminal and run:

```bash
git clone https://github.com/StevenZBX/CS-117-Final-Project.git
cd CS-117-Final-Project
python3 -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell (use python instead of python3 when creating the environment)
.\.venv\Scripts\Activate.ps1
```

From the project root, install the recorded LightGlue version and dependencies. Skip cloning folders that already exist.

```bash
git clone https://github.com/cvg/LightGlue.git third_party/LightGlue
git -C third_party/LightGlue checkout 5a650c444718caffc2669dc0094cde908f42dd73
python -m pip install -r requirements.txt
```

The dataset is downloaded automatically if `data/hpatches-sequences-release/` is absent. Initial setup and model-weight downloads require internet access. Devices are selected automatically in the order CUDA → MPS → CPU; use `--device cuda`, `--device mps`, or `--device cpu` to choose explicitly.

# How to get results

Run both commands from the project root:

```bash
python scripts/train.py
python scripts/evaluate.py
```

Both default to A, B, and C. To run a subset, add the same selection to both commands, for example `--methods A C`.

- `train.py` compares three parameter settings per method using only the 96 train sequences. It saves each method's best parameters to `configs/best_parameters/A.csv`, `B.csv`, and `C.csv`, and the best candidate's 480 pair scores to `results/train_A.csv`, `train_B.csv`, and `train_C.csv`.
- `evaluate.py` loads those saved parameters and uses only the 20 test sequences. It saves 100 pair scores per method to `results/test_A.csv`, `test_B.csv`, and `test_C.csv`.
- Successful re-runs overwrite the corresponding outputs. Training here means parameter selection, not training neural-network weights.

Open `demo.ipynb` with the project's Python environment selected as its kernel and run the cells in order. It displays feature/match examples and comparisons from the train/test CSVs inline, without saving image files; if its dependency cell requests a kernel restart, restart and run from the top again. The notebook also includes Colab setup and automatic device detection.
