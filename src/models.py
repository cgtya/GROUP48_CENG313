"""
Baseline models
"""

from typing import Dict, List, Optional

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
        self.model = Ridge(alpha=alpha, solver="cholesky", random_state=42)
        self.priors_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray | sparse.spmatrix, Y: np.ndarray) -> "RidgeMultilabelBaseline":
        self.priors_ = np.mean(Y, axis=0).astype(np.float32)
        X_dense = X.toarray().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
        self.model.fit(X_dense, Y)
        return self

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        X_dense = X.toarray().astype(np.float32) if sparse.issparse(X) else np.asarray(X, dtype=np.float32)
        raw_scores = self.model.predict(X_dense).astype(np.float32)
        return np.clip(raw_scores, 0.0, 1.0)


if TORCH_AVAILABLE:

    class _MLPNet(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.3) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class TorchMLPBaseline:
    """
    Model 2: PyTorch Multi-Layer Perceptron (MLP) Baseline.
    Automatically utilizes cuda device when available,
    or falls back to CPU.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 12,
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TorchMLPBaseline.")
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_: Optional[_MLPNet] = None
        self.history_: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    @staticmethod
    def _to_dense_float32(X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        if sparse.issparse(X):
            return X.toarray().astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def fit(
        self,
        X_train: np.ndarray | sparse.spmatrix,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray | sparse.spmatrix] = None,
        Y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> "TorchMLPBaseline":
        torch.manual_seed(42)
        X_tr = self._to_dense_float32(X_train)
        Y_tr = np.asarray(Y_train, dtype=np.float32)

        input_dim = X_tr.shape[1]
        output_dim = Y_tr.shape[1]

        self.model_ = _MLPNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(Y_tr))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.history_ = {"train_loss": [], "val_loss": []}

        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            running_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * xb.size(0)

            epoch_train_loss = running_loss / len(dataset)
            self.history_["train_loss"].append(epoch_train_loss)

            if X_val is not None and Y_val is not None:
                val_loss = self._eval_loss(X_val, Y_val, criterion)
                self.history_["val_loss"].append(val_loss)
                if verbose and (epoch == 1 or epoch % 3 == 0 or epoch == self.epochs):
                    print(
                        f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                        f"train_loss={epoch_train_loss:.4f} | val_loss={val_loss:.4f} "
                        f"(device={self.device})"
                    )
            elif verbose and (epoch == 1 or epoch % 3 == 0 or epoch == self.epochs):
                print(
                    f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                    f"train_loss={epoch_train_loss:.4f} (device={self.device})"
                )

        return self

    def _eval_loss(
        self,
        X_val: np.ndarray | sparse.spmatrix,
        Y_val: np.ndarray,
        criterion: nn.Module,
    ) -> float:
        assert self.model_ is not None
        self.model_.eval()
        X_v = self._to_dense_float32(X_val)
        Y_v = np.asarray(Y_val, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X_v), torch.from_numpy(Y_v))
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model_(xb)
                total_loss += criterion(logits, yb).item() * xb.size(0)
        return total_loss / len(ds)

    def predict_proba(self, X: np.ndarray | sparse.spmatrix) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("TorchMLPBaseline must be fitted before calling predict_proba.")
        self.model_.eval()
        X_dense = self._to_dense_float32(X)
        ds = TensorDataset(torch.from_numpy(X_dense))
        loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)

        preds: List[np.ndarray] = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                probs = torch.sigmoid(self.model_(xb)).cpu().numpy()
                preds.append(probs)

        return np.vstack(preds).astype(np.float32)
