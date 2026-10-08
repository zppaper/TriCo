from typing import Dict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
    }


def selection_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    result = metrics(labels, probabilities)
    return (result["auc"] + result["aupr"]) / 2.0
