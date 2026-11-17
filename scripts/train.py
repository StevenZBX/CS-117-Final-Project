"""Select A/B/C matching parameters on the 96 train sequences only.

This is parameter search for pretrained methods, not neural-weight training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from dataset_load import ROOT, get_data_root, validate_splits
from evaluate import (MODULES, PARAMETER_FIELDS, config_id, train_manifest_id,
                      validate_config, run_pairs, write_results, write_csv)

CANDIDATES = {"A": (0.7, 0.8, 0.9), "B": (0.1, 0.2, 0.3), "C": (0.1, 0.2, 0.3)}


def train(args):
    config = json.loads(args.config.read_text())
    validate_config(config, args.methods)
    data = get_data_root(args.data)
    ids, _ = validate_splits(data, "train")
    manifest_id = train_manifest_id()
    print(f"train only: {len(ids)} sequences / {len(ids)*5} pairs per candidate")
    saved = {}
    for name in args.methods:
        parameter = "ratio" if name == "A" else "filter_threshold"
        best = None
        for value in CANDIDATES[name]:
            print(f"{name}: trying {parameter}={value}", flush=True)
            rows = run_pairs(name, {**config, parameter: value}, data, ids, args.cache, args.device)
            candidate_path = ROOT / "data/train_candidates" / manifest_id[:16] / config_id(config) / name / f"{value}.csv"
            write_results(candidate_path, rows)
            mean = float(np.mean([row["mma@3"] for row in rows]))
            # Strict improvement means ties retain the first candidate.
            if best is None or mean > best[0]:
                best = mean, value, rows
        mean, value, rows = best
        result_path = args.output / f"train_{name}.csv"
        parameter_path = args.parameters / f"{name}.csv"
        write_results(result_path, rows)
        write_csv(parameter_path, [{"method": name, "parameter": parameter, "value": value,
                                   "train_mma@3": mean, "config_id": config_id(config),
                                   "train_manifest_id": manifest_id}], PARAMETER_FIELDS)
        saved[name] = parameter_path
        print(f"{name}: best {parameter}={value}, mean train MMA@3={mean:.6f}; saved {parameter_path}")
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=list(MODULES), default=list(MODULES))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data", type=Path, help="Local dataset; otherwise local default or download")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/feature_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--parameters", type=Path, default=ROOT / "configs/best_parameters")
    args = parser.parse_args()
    if len(set(args.methods)) != len(args.methods):
        parser.error("Duplicate methods")
    train(args)


if __name__ == "__main__":
    main()
