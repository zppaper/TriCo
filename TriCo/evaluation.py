import gc
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from . import config
from .model import TriCo
from .training import make_loader, set_seed, train_model
from .metrics import metrics, selection_score


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-05, 1.0 - 1e-05)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(logits: np.ndarray) -> np.ndarray:
    bounded = np.clip(logits, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def fit_modality_expert(
    train_values: np.ndarray,
    y_train: np.ndarray,
    val_values: np.ndarray,
    y_val: np.ndarray,
    test_values: np.ndarray,
    run_seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    best = None
    for feature_count in config.EXPERT_FEATURE_GRID:
        usable_features = min(feature_count, train_values.shape[1])
        for c_value in config.EXPERT_C_GRID:
            expert = make_pipeline(
                SelectKBest(f_classif, k=usable_features),
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=run_seed,
                ),
            )
            expert.fit(train_values, y_train)
            val_probabilities = expert.predict_proba(val_values)[:, 1]
            candidate = (
                selection_score(y_val, val_probabilities),
                -feature_count,
                -c_value,
                expert,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("Modality expert selection produced no model")
    (score, _, _, expert) = best
    return (
        expert.predict_proba(val_values)[:, 1],
        expert.predict_proba(test_values)[:, 1],
        float(score),
    )


def raw_reliability_weights(scores: Dict[str, float]) -> Dict[str, float]:
    values = np.asarray([scores[name] for name in config.MODALITY_NAMES])
    soft = np.exp((values - values.max()) / config.RELIABILITY_TEMPERATURE)
    soft /= soft.sum()
    return dict(zip(config.MODALITY_NAMES, soft.tolist()))


def reliability_weights(scores: Dict[str, float]) -> Dict[str, float]:
    raw = raw_reliability_weights(scores)
    soft = np.asarray([raw[name] for name in config.MODALITY_NAMES])
    weights = (
        1.0 - config.UNIFORM_WEIGHT_PRIOR
    ) * soft + config.UNIFORM_WEIGHT_PRIOR / len(config.MODALITY_NAMES)
    return dict(zip(config.MODALITY_NAMES, weights.tolist()))


def modality_weight_imbalance(weights: Dict[str, float]) -> float:
    values = np.asarray([weights[name] for name in config.MODALITY_NAMES])
    return float(values.std(ddof=0) / values.mean())


def extract_latent_evidence(
    model: nn.Module, loader: DataLoader
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (labels, primary_logits) = ([], [])
    features = {name: [] for name in config.MODALITY_NAMES}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            (mrna, mirna_gene, cnv, mirna_raw, targets) = [
                tensor.to(config.DEVICE) for tensor in batch
            ]
            outputs = model(mrna, mirna_gene, cnv, mirna_raw)
            ru = outputs["ru"]
            for name in config.MODALITY_NAMES:
                features[name].append(
                    torch.cat(
                        [ru["shared_candidates"][name], ru["z_unique"][name]], dim=1
                    )
                    .cpu()
                    .numpy()
                )
            primary_logits.append(outputs["logits"].cpu().numpy())
            labels.append(targets.cpu().numpy())
    joint_features = np.concatenate(
        [np.concatenate(features[name]) for name in config.MODALITY_NAMES], axis=1
    )
    return (
        np.concatenate(labels).astype(int),
        np.concatenate(primary_logits),
        joint_features,
    )


def fit_joint_conflict_head(
    train_values: np.ndarray,
    y_train: np.ndarray,
    val_values: np.ndarray,
    y_val: np.ndarray,
    test_values: np.ndarray,
    run_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    best = None
    for feature_count in config.JOINT_FEATURE_GRID:
        usable_features = min(feature_count, train_values.shape[1])
        for c_value in config.JOINT_C_GRID:
            head = make_pipeline(
                SelectKBest(f_classif, k=usable_features),
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=run_seed,
                ),
            )
            head.fit(train_values, y_train)
            val_probabilities = head.predict_proba(val_values)[:, 1]
            candidate = (
                selection_score(y_val, val_probabilities),
                -feature_count,
                -c_value,
                head,
                val_probabilities,
            )
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("Joint conflict-head selection produced no model")
    return (best[4], best[3].predict_proba(test_values)[:, 1])


def fuse_modalities(
    primary_logits: np.ndarray,
    modality_probabilities: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    balanced_logits = sum(
        (
            weights[name] * logit(modality_probabilities[name])
            for name in config.MODALITY_NAMES
        )
    )
    return (
        1.0 - config.MODALITY_FUSION_WEIGHT
    ) * primary_logits + config.MODALITY_FUSION_WEIGHT * balanced_logits


def conflict_residual(
    primary_logits: np.ndarray, joint_probabilities: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_logits = logit(joint_probabilities)
    direction_conflict = np.sign(primary_logits) != np.sign(joint_logits)
    excess = np.clip(
        np.abs(primary_logits - joint_logits) - config.CONFLICT_MARGIN, 0.0, 1.0
    )
    residual = (
        config.CONFLICT_CORRECTION_SCALE
        * direction_conflict
        * excess
        * (joint_logits - primary_logits)
    )
    active_conflict = direction_conflict & (excess > 0.0)
    return (residual, active_conflict, joint_logits)


def run_model(
    train_arrays: Sequence[np.ndarray],
    val_arrays: Sequence[np.ndarray],
    test_arrays: Sequence[np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    run_seed: int,
    gene_pathway: pd.DataFrame,
    relation_specs: Dict[str, Dict[str, object]],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    set_seed(run_seed)
    train_loader = make_loader(train_arrays, y_train, False, run_seed)
    shuffled_loader = make_loader(train_arrays, y_train, True, run_seed)
    val_loader = make_loader(val_arrays, y_val, False, run_seed)
    test_loader = make_loader(test_arrays, y_test, False, run_seed)
    model = TriCo(gene_pathway=gene_pathway, relation_specs=relation_specs).to(
        config.DEVICE
    )
    train_model(model, shuffled_loader, val_loader, y_train)
    (train_labels, _, train_joint) = extract_latent_evidence(model, train_loader)
    (val_labels, val_primary, val_joint) = extract_latent_evidence(model, val_loader)
    (test_labels, test_primary, test_joint) = extract_latent_evidence(
        model, test_loader
    )
    if not np.array_equal(train_labels, y_train):
        raise RuntimeError("Training representation order changed")
    if not np.array_equal(val_labels, y_val) or not np.array_equal(test_labels, y_test):
        raise RuntimeError("Evaluation representation order changed")
    (_, joint_test) = fit_joint_conflict_head(
        train_joint, y_train, val_joint, y_val, test_joint, run_seed
    )
    (val_modality, test_modality, scores) = ({}, {}, {})
    for (offset, name) in enumerate(config.MODALITY_NAMES):
        array_index = config.MODALITY_ARRAY_INDEX[name]
        (val_probabilities, test_probabilities, score) = fit_modality_expert(
            train_arrays[array_index],
            y_train,
            val_arrays[array_index],
            y_val,
            test_arrays[array_index],
            run_seed + offset,
        )
        val_modality[name] = val_probabilities
        test_modality[name] = test_probabilities
        scores[name] = score
    raw_weights = raw_reliability_weights(scores)
    weights = reliability_weights(scores)
    imbalance_before = modality_weight_imbalance(raw_weights)
    imbalance_after = modality_weight_imbalance(weights)
    imbalance_reduction = (
        (imbalance_before - imbalance_after) / imbalance_before
        if imbalance_before > 0.0
        else 0.0
    )
    fuse_modalities(val_primary, val_modality, weights)
    baseline_logits = fuse_modalities(test_primary, test_modality, weights)
    (residual, _, _) = conflict_residual(test_primary, joint_test)
    final_logits = baseline_logits + residual
    final_probabilities = sigmoid(final_logits)
    result = {
        **metrics(y_test, final_probabilities),
        "imbalance_before": imbalance_before,
        "imbalance_after": imbalance_after,
        "imbalance_reduction": float(imbalance_reduction),
        "max_weight_before": max(raw_weights.values()),
        "max_weight_after": max(weights.values()),
        **{f"raw_weight_{name}": raw_weights[name] for name in config.MODALITY_NAMES},
        **{f"weight_{name}": weights[name] for name in config.MODALITY_NAMES},
    }
    predictions = [
        {
            "sample": sample,
            "label": int(y_test[sample]),
            "probability": float(final_probabilities[sample]),
        }
        for sample in range(len(y_test))
    ]
    del model, train_loader, shuffled_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (result, predictions)
