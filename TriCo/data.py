import csv
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config


def clean_mirna_name(name: str) -> str:
    value = str(name).lower().strip().replace("_", "-").replace("mirna", "mir")
    return re.sub("\\*$", "", value)


def mirna_aliases(name: str) -> Set[str]:
    cleaned = clean_mirna_name(name)
    aliases = {cleaned, cleaned[4:] if cleaned.startswith("hsa-") else f"hsa-{cleaned}"}
    expanded = set(aliases)
    for alias in aliases:
        no_arm = re.sub("-(3|5)p$", "", alias)
        expanded.add(no_arm)
        expanded.add(no_arm[4:] if no_arm.startswith("hsa-") else f"hsa-{no_arm}")
    return {alias for alias in expanded if alias}


def parse_gmt(path: Path) -> Dict[str, List[str]]:
    pathways: Dict[str, List[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = [
                g.strip().upper()
                for g in parts[2:]
                if g.strip() and g.strip().upper() != "NA"
            ]
            if genes:
                pathways[parts[0].split("_")[0].strip()] = genes
    if not pathways:
        raise ValueError(f"No pathways parsed from {path}")
    return pathways


def load_anchor_genes(gene_level_dir: Path) -> List[str]:
    anchor_file = gene_level_dir / "genes_anchor.txt"
    if anchor_file.exists():
        genes = [
            line.strip().upper()
            for line in anchor_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        genes = sorted(
            {
                gene
                for members in parse_gmt(config.PATHWAY_FILE).values()
                for gene in members
            }
        )
    if not genes:
        raise ValueError("Anchor gene list is empty")
    return genes


def build_gene_pathway_mask(anchor_genes: Sequence[str]) -> pd.DataFrame:
    gene_index = {gene: idx for (idx, gene) in enumerate(anchor_genes)}
    (columns, names) = ([], [])
    for (pathway, members) in sorted(parse_gmt(config.PATHWAY_FILE).items()):
        indices = [gene_index[gene] for gene in members if gene in gene_index]
        if not indices:
            continue
        column = np.zeros(len(anchor_genes), dtype=np.float32)
        column[indices] = 1.0
        columns.append(column)
        names.append(pathway)
    if not columns:
        raise ValueError("No KEGG pathway overlaps the anchor genes")
    return pd.DataFrame(np.stack(columns, axis=1), index=anchor_genes, columns=names)


@lru_cache(maxsize=2)
def load_mirna_targets(path_string: str) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = defaultdict(set)
    with Path(path_string).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        if {"source", "target", "relation_type"}.issubset(fields):
            (mirna_col, gene_col, type_col) = ("source", "target", "relation_type")
        elif "miRNA" in fields:
            mirna_col = "miRNA"
            gene_col = "target_gene" if "target_gene" in fields else "Target Gene"
            type_col = None
        else:
            raise ValueError(f"Unsupported miRNA target columns in {path_string}")
        for row in reader:
            if type_col and row[type_col] != "MIR_REPRESSION":
                continue
            gene = str(row[gene_col]).strip().upper()
            if not gene:
                continue
            for alias in mirna_aliases(row[mirna_col]):
                result[alias].add(gene)
    return dict(result)


def read_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0).T
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    frame.index = frame.index.astype(str).str.upper()
    frame.columns = frame.columns.astype(str)
    return frame


def project_gene_matrix(
    frame: pd.DataFrame, anchor_genes: Sequence[str]
) -> pd.DataFrame:
    frame.columns = frame.columns.astype(str).str.upper()
    return frame.reindex(columns=anchor_genes, fill_value=0.0).astype(np.float32)


def project_mirna_to_genes(
    mirna: pd.DataFrame, anchor_genes: Sequence[str]
) -> pd.DataFrame:
    target_map = load_mirna_targets(str(config.MIRNA_TARGET_FILE))
    gene_index = {gene: idx for (idx, gene) in enumerate(anchor_genes)}
    association = np.zeros((mirna.shape[1], len(anchor_genes)), dtype=np.float32)
    for (mirna_idx, feature) in enumerate(mirna.columns):
        targets: Set[str] = set()
        for alias in mirna_aliases(feature):
            targets.update(target_map.get(alias, set()))
        indices = [gene_index[gene] for gene in targets if gene in gene_index]
        association[mirna_idx, indices] = 1.0
    degree = association.sum(axis=0)
    supported = degree > 0
    association[:, supported] /= degree[supported]
    values = mirna.to_numpy(np.float32) @ association
    return pd.DataFrame(values, index=mirna.index, columns=anchor_genes)


def load_dataset(
    dataset: str, base_dir: Path, gene_level_dir: Path, anchor_genes: Sequence[str]
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    raw_dir = base_dir / dataset
    gene_dir = gene_level_dir / dataset
    required = [
        raw_dir / "mRNA.csv",
        raw_dir / "miRNA.csv",
        raw_dir / "label.csv",
        gene_dir / "cnv_gene.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {dataset}: {missing}")
    mrna_raw = read_matrix(raw_dir / "mRNA.csv")
    mirna_raw = read_matrix(raw_dir / "miRNA.csv")
    cnv_raw = read_matrix(gene_dir / "cnv_gene.csv")
    labels = pd.read_csv(raw_dir / "label.csv")
    if "sample_id" in labels.columns:
        labels = labels.set_index("sample_id")
    if "label" not in labels.columns:
        raise ValueError(f"label column missing from {raw_dir / 'label.csv'}")
    labels.index = labels.index.astype(str).str.upper()
    mrna = project_gene_matrix(mrna_raw, anchor_genes)
    mirna_gene = project_mirna_to_genes(mirna_raw, anchor_genes)
    cnv = project_gene_matrix(cnv_raw, anchor_genes)
    common = (
        mrna.index.intersection(mirna_gene.index)
        .intersection(cnv.index)
        .intersection(mirna_raw.index)
        .intersection(labels.index)
    )
    if common.empty:
        raise ValueError(f"No aligned samples for {dataset}")
    arrays = [
        mrna.loc[common].to_numpy(np.float32),
        mirna_gene.loc[common].to_numpy(np.float32),
        cnv.loc[common].to_numpy(np.float32),
        mirna_raw.loc[common].to_numpy(np.float32),
    ]
    y = labels.loc[common, "label"].astype(int).to_numpy()
    return (arrays, y, list(mirna_raw.columns))


def split_and_scale(
    arrays: Sequence[np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_seed: int,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    y_train_full = y[train_idx]
    counts = np.bincount(y_train_full.astype(int), minlength=2)
    if counts.min() < 2:
        raise ValueError(f"Insufficient training class counts: {counts.tolist()}")
    relative_indices = np.arange(len(train_idx))
    (sub_train, val) = train_test_split(
        relative_indices,
        test_size=max(2, int(np.ceil(len(relative_indices) * config.VAL_RATIO))),
        stratify=y_train_full,
        random_state=split_seed,
    )
    (train_arrays, val_arrays, test_arrays) = ([], [], [])
    for array in arrays:
        train_full = array[train_idx]
        scaler = StandardScaler().fit(train_full[sub_train])
        train_arrays.append(scaler.transform(train_full[sub_train]).astype(np.float32))
        val_arrays.append(scaler.transform(train_full[val]).astype(np.float32))
        test_arrays.append(scaler.transform(array[test_idx]).astype(np.float32))
    return (
        train_arrays,
        val_arrays,
        test_arrays,
        y_train_full[sub_train],
        y_train_full[val],
        y[test_idx],
    )


def validate_inputs() -> None:
    required = [config.PATHWAY_FILE]
    required.extend((path for (path, _) in config.RELATION_FILES.values()))
    required.extend(
        (config.DATASETS_ROOT / config.TASK / dataset for dataset in config.DATASETS)
    )
    required.extend(
        (
            config.DATASETS_ROOT / f"{config.TASK}_gene_level" / dataset
            for dataset in config.DATASETS
        )
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing experiment inputs: {missing}")
