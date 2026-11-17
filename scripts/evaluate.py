"""Test-only A/B/C evaluation and shared matching/CSV/plot utilities."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import time
import tempfile
import cv2
import numpy as np
from dataset_load import ROOT, get_data_root, validate_splits, load_pairs, errors, score

MODULES = {"A": "method_A", "B": "method_B", "C": "method_C"}
FIELDS = ["sequence", "category", "k", "num_matches"] + [f"mma@{t}" for t in range(1, 11)]
PARAMETER_FIELDS = ["method", "parameter", "value", "train_mma@3", "config_id", "train_manifest_id"]


def build_backends(names, config, cache, device, method_configs=None):
    if not names or len(set(names)) != len(names):
        raise ValueError("Select each method at most once")
    backends = {}
    shared_sift = None
    for name in names:
        if name not in MODULES:
            raise ValueError(f"Unsupported method: {name}")
        module = importlib.import_module(MODULES[name])
        conf = (method_configs or {}).get(name, config)
        if name == "A":
            chosen = "cpu"
        else:
            import torch
            torch.set_num_threads(config["opencv_threads"])
            chosen = module.resolve_device(device)
        extractor, matcher = module.build_model(chosen, conf, cache)
        if name in {"A", "B"}:
            if shared_sift is None:
                shared_sift = extractor
            else:
                extractor = shared_sift
        backends[name] = module, extractor, matcher, chosen
    return backends


def match_points(backend, image0, image1):
    module, extractor, matcher, device = backend
    p0, p1 = module.match(extractor, matcher, image0, image1, device)
    p0, p1 = np.asarray(p0, dtype=np.float64), np.asarray(p1, dtype=np.float64)
    if p0.ndim != 2 or p0.shape[1:] != (2,) or p1.shape != p0.shape:
        raise ValueError("match() must return two equally sized (N,2) point arrays")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("Non-finite match coordinates")
    return p0, p1


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_results(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path}; expected {FIELDS}")
        rows = list(reader)
    for row in rows:
        for key in ("k", "num_matches"):
            row[key] = int(row[key])
        for t in range(1, 11):
            row[f"mma@{t}"] = float(row[f"mma@{t}"])
    return rows


def write_csv(path, rows, fields):
    """Write C-compatible columns; replace an existing CSV only after writing succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_results(path, rows):
    write_csv(path, rows, FIELDS)


def validate_config(config, methods):
    if (config["thresholds"] != list(range(1, 11)) or config["pixel_coordinates"] != "resized"
            or config["zero_matches"] != "zero" or config["threshold_comparison"] != "strict_less_than"):
        raise ValueError("Unsupported scoring configuration")
    if config["max_side"] <= 0 or config["max_keypoints"] <= 0 or not 0 < config["ratio"] < 1:
        raise ValueError("Invalid feature configuration")
    if "B" in methods and config["descriptor_normalization"] != "rootsift":
        raise ValueError("A/B must share RootSIFT features for pretrained LightGlue")


def train_manifest_id():
    return hashlib.sha256((ROOT / "splits/hpatches_train.txt").read_bytes()).hexdigest()


def load_best_config(method, config, parameters):
    path = Path(parameters) / f"{method}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing best parameters: {path}. Run python scripts/train.py --methods {method}, or use --use-defaults explicitly.")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PARAMETER_FIELDS:
            raise ValueError(f"Invalid parameter columns: {path}")
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one best-parameter record: {path}")
    row = rows[0]
    parameter = "ratio" if method == "A" else "filter_threshold"
    if (row["method"] != method or row["parameter"] != parameter
            or row["config_id"] != config_id(config) or row["train_manifest_id"] != train_manifest_id()):
        raise ValueError(f"Saved parameters do not match current configuration/train manifest: {path}. Rerun train.py.")
    value, accuracy = float(row["value"]), float(row["train_mma@3"])
    if not np.isfinite(value) or not 0 < value < 1 or not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError(f"Invalid best-parameter values: {path}")
    print(f"{method}: saved {parameter}={value}, train MMA@3={accuracy:.6f}")
    return {**config, parameter: value}


def run_pairs(name, config, data, ids, cache, device):
    """Common pair runner; callers supply their validated, fixed split IDs."""
    cv2.setNumThreads(config["opencv_threads"])
    cv2.setRNGSeed(0)
    backend = build_backends([name], config, cache, device)[name]
    rows = []
    started = time.perf_counter()
    for i, sequence in enumerate(ids):
        for k in range(2, 7):
            image0, image1, H = load_pairs(data / sequence, k, config["max_side"])
            p0, p1 = match_points(backend, image0, image1)
            err = errors(H, p0, p1)
            rows.append({"sequence": sequence,
                         "category": "illumination" if sequence.startswith("i_") else "viewpoint",
                         "k": k, "num_matches": len(p0),
                         **{f"mma@{t}": score(err, t) for t in range(1, 11)}})
        if hasattr(backend[1], "clear_memory"):
            backend[1].clear_memory()
        print(f"{name} [{i+1}/{len(ids)}] {sequence} {time.perf_counter()-started:.1f}s", flush=True)
    validate_results({name: rows})
    return rows


def evaluate(args):
    config = json.loads(args.config.read_text())
    validate_config(config, args.methods)
    data = get_data_root(args.data)
    ids, _ = validate_splits(data, "test")
    # Load all requested settings before beginning model inference.
    settings = {name: dict(config) if args.use_defaults else load_best_config(name, config, args.parameters)
                for name in args.methods}
    print(f"test only: {len(ids)} sequences / {len(ids)*5} pairs per method")
    paths = {}
    for name in args.methods:
        rows = run_pairs(name, settings[name], data, ids, args.cache, args.device)
        paths[name] = args.output / f"test_{name}.csv"
        write_results(paths[name], rows)
        print(f"Saved {paths[name]}")
    return paths


def main(default_methods=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=list(MODULES), default=default_methods or list(MODULES))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data", type=Path, help="Local dataset; otherwise local default or download")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/feature_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--parameters", type=Path, default=ROOT / "configs/best_parameters")
    parser.add_argument("--use-defaults", action="store_true", help="Explicitly use the base configuration instead of saved train parameters")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods")
    evaluate(args)


LABELS={'A':'SIFT + ratio test','B':'SIFT + LightGlue','C':'SuperPoint + LightGlue'}
COLORS={'A':'#2463a6','B':'#299451','C':'#d87920'}


def validate_results(results):
    if not results:raise ValueError('No results to plot')
    expected=None
    for method,rows in results.items():
        if method not in LABELS or not rows:raise ValueError(f'Unsupported or empty method: {method}')
        keys=set()
        for row in rows:
            key=(row['sequence'],int(row['k']))
            if key in keys:raise ValueError(f'Duplicate pair for {method}: {key}')
            keys.add(key)
            category='illumination' if row['sequence'].startswith('i_') else 'viewpoint'
            if row['category'] != category or int(row['k']) not in range(2,7):
                raise ValueError(f'Invalid category/pair: {key}')
            if row['num_matches'] < 0:raise ValueError('Negative match count')
            values=[float(row[f'mma@{t}']) for t in range(1,11)]
            if not all(np.isfinite(v) and 0<=v<=1 for v in values) or values != sorted(values):
                raise ValueError('MMA must be finite, within [0,1], and monotone')
        if expected is not None and keys != expected:
            raise ValueError('Cannot compare methods evaluated on different image pairs')
        expected=keys


def plot_results(results, figure=None, split="test"):
    import matplotlib.pyplot as plt
    validate_results(results)
    methods=[m for m in LABELS if m in results]
    fig,axes=plt.subplots(3,2,figsize=(12,11))
    for j,category in enumerate(['illumination','viewpoint']):
        mean_matches=[]
        for method in methods:
            sub=[r for r in results[method] if r['category']==category]
            if not sub:continue
            ks=sorted(set(int(r['k']) for r in sub))
            mean3=np.mean([r['mma@3'] for r in sub])
            axes[0,j].plot(ks,[np.mean([r['mma@3'] for r in sub if int(r['k'])==k]) for k in ks],
                           'o-',color=COLORS[method],label=f'{method}: {LABELS[method]} (mean {mean3:.4f})')
            axes[1,j].plot(range(1,11),[np.mean([r[f'mma@{t}'] for r in sub]) for t in range(1,11)],
                           'o-',color=COLORS[method],label=method)
            mean_matches.append((method,float(np.mean([r['num_matches'] for r in sub]))))
        axes[0,j].set(title=category.capitalize(),xlabel='Target image index k (pair 1 -> k)',xticks=range(2,7),ylim=(0,1))
        axes[1,j].set(xlabel='Threshold (resized-image pixels)',ylim=(0,1))
        if mean_matches:
            bars=axes[2,j].bar([m for m,_ in mean_matches],[v for _,v in mean_matches],color=[COLORS[m] for m,_ in mean_matches],width=.55)
            axes[2,j].bar_label(bars,fmt='%.1f',padding=3)
            axes[2,j].margins(y=.2)
            axes[0,j].legend(fontsize=8)
        axes[2,j].set(xlabel='Method',ylabel='Mean returned matches')
    axes[0,0].set_ylabel('MMA@3px');axes[1,0].set_ylabel('MMA')
    for ax in axes.flat:ax.grid(axis='y',alpha=.25)
    pairs=len(next(iter(results.values())))
    fig.suptitle(f'HPatches {split} | {" / ".join(methods)} | {pairs} pairs per method',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.97))
    if figure is not None:
        fig.savefig(figure,dpi=160)
        plt.close(fig)
    return fig



if __name__ == "__main__":
    main()
