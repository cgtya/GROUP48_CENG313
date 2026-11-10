"""
Baseline models
"""

from typing import Optional

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NaiveFrequencyBaseline:
    """
    Model 0 Naive Prior Probability Baseline.
    Assigns the empirical frequency of each GO term in the training set to every protein.
    Serves as the lower bound benchmark required to validate whether sequence features add signal.
    """

    def __init__(self) -> None:
        self.priors_: Optional[np.ndarray] = None

    def fit(self, X: Optional[np.ndarray | sparse.spmatrix], Y: np.ndarray) -> "NaiveFrequencyBaseline":
        self.priors_ = np.mean(Y, axis=0).astype(np.float32)
        return self

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        if self.priors_ is None:
            raise RuntimeError("NaiveFrequencyBaseline must be fitted before calling predict_proba.")
        n_samples = X.shape[0]
        return np.tile(self.priors_, (n_samples, 1))


class RidgeMultilabelBaseline:
    """
    Model 1 L2-regularized Ridge Regression
    solved across all top K GO terms, followed by
    min-max/sigmoid probability calibration.
    """

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, solver="lsqr", max_iter=500, random_state=42)
        self.priors_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray | sparse.spmatrix, Y: np.ndarray) -> "RidgeMultilabelBaseline":
        self.priors_ = np.mean(Y, axis=0).astype(np.float32)
        self.model.fit(X, Y)
        return self

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        raw_scores = self.model.predict(X).astype(np.float32)
        return np.clip(raw_scores, 0.0, 1.0)
