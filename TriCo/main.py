import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
from sklearn.model_selection import StratifiedKFold
from . import config
from .data import (
    build_gene_pathway_mask,
    load_anchor_genes,
    load_dataset,
    split_and_scale,
    validate_inputs,
)
from .relations import build_relation_specs
from .evaluation import run_model
from .results import summarize, write_outputs


def configure_local_run(args: argparse.Namespace) -> None:
    config.EPOCHS = args.epochs
    config.PATIENCE = args.patience
    config.BATCH_SIZE = args.batch_size
    if args.data_dir is not None:
        config.DATASETS_ROOT = args.data_dir.resolve()
    if args.result_dir is not None:
        config.RESULT_DIR = args.result_dir.resolve()
        config.RUN_FILE = config.RESULT_DIR / "runs.csv"
        config.SUMMARY_FILE = config.RESULT_DIR / "summary.csv"
        config.PREDICTION_FILE = config.RESULT_DIR / "predictions.csv"
        config.BALANCE_FILE = config.RESULT_DIR / "modality_balance_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TriCo cancer stage prediction")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    parser.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    parser.add_argument(
        "--folds", nargs="+", type=int, default=list(range(1, config.N_FOLDS + 1))
    )
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    unknown_seeds = sorted(set(args.seeds) - set(config.SEEDS))
    unknown_datasets = sorted(set(args.datasets) - set(config.DATASETS))
    if unknown_seeds or unknown_datasets:
        raise ValueError(
            f"Unsupported seeds={unknown_seeds}, datasets={unknown_datasets}"
        )
    if any((fold < 1 or fold > config.N_FOLDS for fold in args.folds)):
        raise ValueError(f"Folds must be between 1 and {config.N_FOLDS}")
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise ValueError("epochs, patience, and batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    configure_local_run(args)
    validate_inputs()
    base_dir = config.DATASETS_ROOT / config.TASK
    gene_level_dir = config.DATASETS_ROOT / f"{config.TASK}_gene_level"
    anchor_genes = load_anchor_genes(gene_level_dir)
    gene_pathway = build_gene_pathway_mask(anchor_genes)
    run_rows: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    print(f"Device: {config.DEVICE}", flush=True)
    print("TriCo cancer stage prediction", flush=True)
    print(f"Protocol: {len(args.seeds)} seeds x {len(args.folds)} folds", flush=True)
    for dataset in args.datasets:
        (arrays, y, mirna_features) = load_dataset(
            dataset, base_dir, gene_level_dir, anchor_genes
        )
        relation_specs = build_relation_specs(anchor_genes, mirna_features)
        counts = {
            name: (relation_specs[name]["edges"], relation_specs[name]["targets"])
            for name in config.RELATION_NAMES
        }
        print(
            f"\n{dataset}: samples={len(y)}, classes={np.bincount(y, minlength=2).tolist()}, relations={counts}",
            flush=True,
        )
        result_rows: List[Dict[str, float]] = []
        for seed in args.seeds:
            splitter = StratifiedKFold(
                n_splits=config.N_FOLDS, shuffle=True, random_state=seed
            )
            for (fold, (train_idx, test_idx)) in enumerate(
                splitter.split(arrays[0], y), start=1
            ):
                if fold not in args.folds:
                    continue
                split = split_and_scale(arrays, y, train_idx, test_idx, seed + fold)
                (train_arrays, val_arrays, test_arrays, y_train, y_val, y_test) = split
                run_seed = seed * 100 + fold
                (result, predictions) = run_model(
                    train_arrays,
                    val_arrays,
                    test_arrays,
                    y_train,
                    y_val,
                    y_test,
                    run_seed,
                    gene_pathway,
                    relation_specs,
                )
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "fold": fold,
                    "n_test": len(test_idx),
                    **result,
                }
                run_rows.append(row)
                result_rows.append(row)
                prediction_rows.extend(
                    (
                        {"dataset": dataset, "seed": seed, "fold": fold, **prediction}
                        for prediction in predictions
                    )
                )
                print(
                    f"  seed={seed} fold={fold}: AUC/AUPR={result['auc']:.4f}/{result['aupr']:.4f} imbalance={result['imbalance_before']:.4f}->{result['imbalance_after']:.4f}",
                    flush=True,
                )
                write_outputs(run_rows, prediction_rows, summary_rows)
        summary_rows.append(summarize(dataset, result_rows))
        write_outputs(run_rows, prediction_rows, summary_rows)
        print("  summary updated", flush=True)
    print(f"\nCompleted. Saved: {config.SUMMARY_FILE}", flush=True)
