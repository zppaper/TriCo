import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from . import config
from .data import mirna_aliases


@lru_cache(maxsize=8)
def read_relation_edges(path: Path, expected_type: str) -> Tuple[Dict[str, str], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "target", "relation_type", "sign", "confidence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return tuple((row for row in reader if row["relation_type"] == expected_type))


def mirna_alias_index(features: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for (idx, feature) in enumerate(features):
        for alias in mirna_aliases(feature):
            result.setdefault(alias, idx)
    return result


def make_relation_spec(
    rows: Sequence[Dict[str, str]],
    source_index: Dict[str, int],
    target_index: Dict[str, int],
    source_dim: int,
    source_is_mirna: bool,
) -> Dict[str, object]:
    by_target: Dict[int, Dict[int, float]] = defaultdict(dict)
    confidence_by_target: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        target_idx = target_index.get(str(row["target"]).strip().upper())
        if target_idx is None:
            continue
        if source_is_mirna:
            source_idx = next(
                (
                    source_index[a]
                    for a in mirna_aliases(row["source"])
                    if a in source_index
                ),
                None,
            )
        else:
            source_idx = source_index.get(str(row["source"]).strip().upper())
        if source_idx is None:
            continue
        value = float(row["sign"]) * float(row["confidence"])
        by_target[target_idx][source_idx] = (
            by_target[target_idx].get(source_idx, 0.0) + value
        )
        confidence_by_target[target_idx].append(abs(float(row["confidence"])))
    (indices, values) = ([], [])
    target_mask = np.zeros(len(target_index), dtype=np.float32)
    quality = np.zeros(len(target_index), dtype=np.float32)
    for (target_idx, source_weights) in by_target.items():
        denominator = sum((abs(value) for value in source_weights.values()))
        if denominator <= 0:
            continue
        target_mask[target_idx] = 1.0
        quality[target_idx] = float(np.mean(confidence_by_target[target_idx]))
        for (source_idx, value) in source_weights.items():
            indices.append([source_idx, target_idx])
            values.append(value / denominator)
    index_array = (
        np.asarray(indices, dtype=np.int64).T
        if indices
        else np.zeros((2, 0), dtype=np.int64)
    )
    value_array = np.asarray(values, dtype=np.float32)
    return {
        "indices": index_array,
        "values": value_array,
        "shape": (source_dim, len(target_index)),
        "mask": target_mask,
        "quality": quality,
        "edges": len(values),
        "targets": int(target_mask.sum()),
    }


def build_relation_specs(
    anchor_genes: Sequence[str], mirna_features: Sequence[str]
) -> Dict[str, Dict[str, object]]:
    gene_index = {gene: idx for (idx, gene) in enumerate(anchor_genes)}
    mirna_index = mirna_alias_index(mirna_features)
    source_info = {
        "cnv": (gene_index, len(anchor_genes), False),
        "mirna": (mirna_index, len(mirna_features), True),
        "tf": (gene_index, len(anchor_genes), False),
        "ctx": (gene_index, len(anchor_genes), False),
    }
    specs = {}
    for name in config.RELATION_NAMES:
        (path, relation_type) = config.RELATION_FILES[name]
        (index, dimension, is_mirna) = source_info[name]
        specs[name] = make_relation_spec(
            read_relation_edges(path, relation_type),
            index,
            gene_index,
            dimension,
            is_mirna,
        )
        if specs[name]["edges"] == 0:
            raise ValueError(f"No usable {name} edges for the current dataset")
    return specs
