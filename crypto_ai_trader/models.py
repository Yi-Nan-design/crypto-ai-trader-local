from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os
from pathlib import Path
import pickle
import warnings

import numpy as np

from .alpha_models import build_lightgbm_classifier_alpha

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def binary_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_prob)
    sorted_prob = y_prob[order]
    ranks = np.empty(len(y_prob), dtype=float)
    start = 0
    while start < len(y_prob):
        end = start + 1
        while end < len(y_prob) and sorted_prob[end] == sorted_prob[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum_pos = float(ranks[positives].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    accuracy = float((pred == y_true).mean())
    tp = float(((pred == 1) & (y_true == 1)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    specificity = tn / max(tn + fp, 1.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "auc": binary_auc(y_true, y_prob),
        "log_loss": binary_log_loss(y_true, y_prob),
    }


@dataclass
class StandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fitted")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


class LogisticRegressionNumpy:
    name = "logistic_regression_numpy"

    def __init__(self, learning_rate: float = 0.03, epochs: int = 800, l2: float = 1e-3, seed: int = 42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "LogisticRegressionNumpy":
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0, 0.01, size=x.shape[1])
        self.bias = 0.0
        n = len(x)
        weights = np.ones(n, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = np.clip(weights, 0.05, 20.0)
        weight_sum = float(weights.sum()) or float(n)
        for _ in range(self.epochs):
            prob = sigmoid(x @ self.weights + self.bias)
            error = (prob - y) * weights
            grad_w = (x.T @ error) / weight_sum + self.l2 * self.weights
            grad_b = float(error.sum() / weight_sum)
            self.weights -= self.learning_rate * grad_w
            self.bias -= self.learning_rate * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted")
        prob = sigmoid(x @ self.weights + self.bias)
        return np.column_stack([1 - prob, prob])


class MLPClassifierNumpy:
    name = "mlp_numpy"

    def __init__(
        self,
        hidden_size: int = 32,
        learning_rate: float = 0.002,
        epochs: int = 450,
        batch_size: int = 256,
        l2: float = 1e-4,
        seed: int = 42,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.w1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.w2: np.ndarray | None = None
        self.b2: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "MLPClassifierNumpy":
        rng = np.random.default_rng(self.seed)
        n_features = x.shape[1]
        self.w1 = rng.normal(0, np.sqrt(2 / n_features), size=(n_features, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = rng.normal(0, np.sqrt(2 / self.hidden_size), size=(self.hidden_size, 1))
        self.b2 = 0.0

        mw1 = np.zeros_like(self.w1)
        vw1 = np.zeros_like(self.w1)
        mb1 = np.zeros_like(self.b1)
        vb1 = np.zeros_like(self.b1)
        mw2 = np.zeros_like(self.w2)
        vw2 = np.zeros_like(self.w2)
        mb2 = 0.0
        vb2 = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 0

        indices = np.arange(len(x))
        weights = np.ones(len(x), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        weights = np.clip(weights, 0.05, 20.0)
        for _ in range(self.epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = x[batch_idx]
                yb = y[batch_idx].reshape(-1, 1)
                wb = weights[batch_idx].reshape(-1, 1)
                wb = wb / max(float(wb.mean()), 1e-8)

                hidden_linear = xb @ self.w1 + self.b1
                hidden = np.maximum(hidden_linear, 0)
                prob = sigmoid(hidden @ self.w2 + self.b2)
                error = (prob - yb) * wb / len(xb)

                grad_w2 = hidden.T @ error + self.l2 * self.w2
                grad_b2 = float(error.sum())
                grad_hidden = error @ self.w2.T
                grad_hidden[hidden_linear <= 0] = 0
                grad_w1 = xb.T @ grad_hidden + self.l2 * self.w1
                grad_b1 = grad_hidden.sum(axis=0)

                step += 1
                mw1, vw1, self.w1 = self._adam_update(self.w1, grad_w1, mw1, vw1, step, beta1, beta2, eps)
                mb1, vb1, self.b1 = self._adam_update(self.b1, grad_b1, mb1, vb1, step, beta1, beta2, eps)
                mw2, vw2, self.w2 = self._adam_update(self.w2, grad_w2, mw2, vw2, step, beta1, beta2, eps)
                mb2, vb2, b2_array = self._adam_update(
                    np.array([self.b2]),
                    np.array([grad_b2]),
                    np.array([mb2]),
                    np.array([vb2]),
                    step,
                    beta1,
                    beta2,
                    eps,
                )
                mb2 = float(mb2[0])
                vb2 = float(vb2[0])
                self.b2 = float(b2_array[0])
        return self

    def _adam_update(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        step: int,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1**step)
        v_hat = v / (1 - beta2**step)
        param = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)
        return m, v, param

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.w1 is None or self.b1 is None or self.w2 is None:
            raise RuntimeError("Model is not fitted")
        hidden = np.maximum(x @ self.w1 + self.b1, 0)
        prob = sigmoid(hidden @ self.w2 + self.b2).reshape(-1)
        return np.column_stack([1 - prob, prob])


class TorchSequenceClassifierAdapter:
    """Optional lightweight sequence classifier backed by PyTorch.

    The adapter stores only numpy copies of the fitted state dict so pickled
    ModelBundle files do not depend on a locally scoped torch module class.
    """

    def __init__(
        self,
        name: str = "torch_transformer_sequence",
        *,
        sequence_length: int = 24,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.15,
        epochs: int = 28,
        batch_size: int = 128,
        learning_rate: float = 8e-4,
        weight_decay: float = 1e-4,
        patience: int = 4,
        validation_fraction: float = 0.15,
        max_train_rows: int = 9000,
        seed: int = 42,
    ):
        self.name = name
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.max_train_rows = max_train_rows
        self.seed = seed
        self.n_features_: int | None = None
        self.state_dict_: dict[str, np.ndarray] | None = None
        self.training_history_: list[dict[str, float]] = []
        self.best_validation_loss_: float | None = None
        self.fitted_epochs_: int = 0

    def _make_sequences(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n, n_features = x.shape
        sequences = np.zeros((n, self.sequence_length, n_features), dtype=np.float32)
        for idx in range(n):
            start = max(0, idx - self.sequence_length + 1)
            window = x[start : idx + 1]
            sequences[idx, -len(window) :, :] = window
        return sequences

    def _build_model(self, n_features: int):
        import torch
        from torch import nn

        class SequenceTransformer(nn.Module):
            def __init__(self, outer: "TorchSequenceClassifierAdapter"):
                super().__init__()
                self.input_proj = nn.Linear(n_features, outer.d_model)
                self.positional = nn.Parameter(torch.zeros(1, outer.sequence_length, outer.d_model))
                layer = nn.TransformerEncoderLayer(
                    d_model=outer.d_model,
                    nhead=outer.nhead,
                    dim_feedforward=outer.dim_feedforward,
                    dropout=outer.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                try:
                    self.encoder = nn.TransformerEncoder(layer, num_layers=outer.num_layers, enable_nested_tensor=False)
                except TypeError:
                    self.encoder = nn.TransformerEncoder(layer, num_layers=outer.num_layers)
                self.norm = nn.LayerNorm(outer.d_model)
                hidden = max(8, outer.d_model // 2)
                self.head = nn.Sequential(
                    nn.Linear(outer.d_model, hidden),
                    nn.GELU(),
                    nn.Dropout(outer.dropout),
                    nn.Linear(hidden, 1),
                )

            def forward(self, batch):
                encoded = self.input_proj(batch) + self.positional
                encoded = self.encoder(encoded)
                pooled = self.norm(encoded[:, -1, :])
                return self.head(pooled).squeeze(-1)

        return SequenceTransformer(self)

    def _numpy_state(self, model) -> dict[str, np.ndarray]:
        return {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}

    def _load_model(self):
        if self.n_features_ is None or self.state_dict_ is None:
            raise RuntimeError("Torch sequence model is not fitted")
        import torch

        model = self._build_model(self.n_features_)
        state = {key: torch.tensor(value) for key, value in self.state_dict_.items()}
        model.load_state_dict(state)
        model.eval()
        return model

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "TorchSequenceClassifierAdapter":
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("missing_dependency: torch")

        import torch

        torch.manual_seed(self.seed)
        torch.set_num_threads(1)
        self.n_features_ = int(x.shape[1])
        sequences = self._make_sequences(x)
        labels = np.asarray(y, dtype=np.float32)
        weights = np.ones(len(labels), dtype=np.float32) if sample_weight is None else np.asarray(sample_weight, dtype=np.float32)
        weights = np.clip(weights, 0.05, 20.0)

        if len(sequences) > self.max_train_rows:
            sequences = sequences[-self.max_train_rows :]
            labels = labels[-self.max_train_rows :]
            weights = weights[-self.max_train_rows :]

        n_rows = len(sequences)
        if n_rows < max(60, self.sequence_length * 3):
            raise RuntimeError(f"not_enough_rows_for_sequence_model: {n_rows}")

        validation_rows = int(n_rows * self.validation_fraction)
        validation_rows = max(32, validation_rows) if n_rows >= 220 else 0
        train_end = n_rows - validation_rows if validation_rows else n_rows
        train_end = max(32, train_end)

        x_train = torch.tensor(sequences[:train_end], dtype=torch.float32)
        y_train = torch.tensor(labels[:train_end], dtype=torch.float32)
        w_train = torch.tensor(weights[:train_end] / max(float(weights[:train_end].mean()), 1e-8), dtype=torch.float32)
        x_valid = torch.tensor(sequences[train_end:], dtype=torch.float32) if validation_rows else x_train
        y_valid = torch.tensor(labels[train_end:], dtype=torch.float32) if validation_rows else y_train
        w_valid = torch.tensor(weights[train_end:] / max(float(weights[train_end:].mean()), 1e-8), dtype=torch.float32) if validation_rows else w_train

        model = self._build_model(self.n_features_)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        best_loss = float("inf")
        best_state: dict[str, np.ndarray] | None = None
        patience_left = self.patience
        indices = np.arange(train_end)
        rng = np.random.default_rng(self.seed)
        self.training_history_ = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            rng.shuffle(indices)
            train_losses = []
            for start in range(0, train_end, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                xb = x_train[batch_idx]
                yb = y_train[batch_idx]
                wb = w_train[batch_idx]
                optimizer.zero_grad()
                logits = model(xb)
                loss = (loss_fn(logits, yb) * wb).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(float(loss.detach().cpu()))

            model.eval()
            with torch.no_grad():
                valid_logits = model(x_valid)
                valid_loss = float((loss_fn(valid_logits, y_valid) * w_valid).mean().detach().cpu())
            train_loss = float(np.mean(train_losses)) if train_losses else valid_loss
            self.training_history_.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": valid_loss})
            self.fitted_epochs_ = epoch
            if valid_loss < best_loss - 1e-4:
                best_loss = valid_loss
                best_state = self._numpy_state(model)
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.best_validation_loss_ = best_loss
        self.state_dict_ = best_state or self._numpy_state(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError("missing_dependency: torch")

        import torch

        model = self._load_model()
        sequences = self._make_sequences(x)
        probs: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                xb = torch.tensor(sequences[start : start + self.batch_size], dtype=torch.float32)
                prob = torch.sigmoid(model(xb)).detach().cpu().numpy()
                probs.append(prob)
        up = np.concatenate(probs).reshape(-1)
        return np.column_stack([1 - up, up])


class SklearnModelAdapter:
    def __init__(self, name: str, model: object):
        self.name = name
        self.model = model

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "SklearnModelAdapter":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weight is None:
                self.model.fit(x, y)
            else:
                try:
                    self.model.fit(x, y, sample_weight=sample_weight)
                except TypeError:
                    self.model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(self.model, "predict_proba"):
                prob = self.model.predict_proba(x)
                if prob.shape[1] == 2:
                    return prob
                up = prob[:, -1]
                return np.column_stack([1 - up, up])
            score = self.model.decision_function(x)
        up = sigmoid(score)
        return np.column_stack([1 - up, up])


class EnsembleProbabilityModel:
    def __init__(self, name: str, models: list[object], model_names: list[str]):
        self.name = name
        self.models = models
        self.model_names = model_names

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnsembleProbabilityModel":
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Ensemble has no fitted models")
        probs = [model.predict_proba(x)[:, 1] for model in self.models]
        up = np.mean(np.vstack(probs), axis=0)
        return np.column_stack([1 - up, up])


@dataclass
class ModelBundle:
    model_name: str
    model: object
    scaler: StandardScaler
    feature_columns: list[str]
    metrics: dict[str, float]
    config: dict[str, float | int | str]
    auxiliary_models: dict[str, object] = field(default_factory=dict)
    auxiliary_metadata: dict[str, dict[str, object]] = field(default_factory=dict)

    def predict_up_probability(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)[:, 1]

    def predict_direction_probabilities(self, x: np.ndarray) -> dict[str, np.ndarray]:
        x_scaled = self.scaler.transform(x)
        prob_up = self.model.predict_proba(x_scaled)[:, 1]
        auxiliary = getattr(self, "auxiliary_models", {}) or {}
        long_model = auxiliary.get("direction_long")
        short_model = auxiliary.get("direction_short")
        trade_model = auxiliary.get("direction_trade")
        prob_long = long_model.predict_proba(x_scaled)[:, 1] if long_model is not None else prob_up
        prob_short = short_model.predict_proba(x_scaled)[:, 1] if short_model is not None else 1.0 - prob_up
        prob_trade = trade_model.predict_proba(x_scaled)[:, 1] if trade_model is not None else np.ones(len(prob_up), dtype=float)
        return {
            "up": prob_up,
            "long": np.asarray(prob_long, dtype=float),
            "short": np.asarray(prob_short, dtype=float),
            "trade": np.asarray(prob_trade, dtype=float),
            "uses_directional_models": np.full(len(prob_up), bool(long_model is not None and short_model is not None), dtype=bool),
            "uses_tradeability_model": np.full(len(prob_up), bool(trade_model is not None), dtype=bool),
        }

    def predict_shadow_direction_probabilities(
        self,
        x: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Return probabilities from validation-qualified shadow side models."""

        x_scaled = self.scaler.transform(x)
        auxiliary = getattr(self, "auxiliary_models", {}) or {}
        long_model = auxiliary.get("shadow_direction_long")
        short_model = auxiliary.get("shadow_direction_short")
        rows = len(x_scaled)
        return {
            "long": (
                np.asarray(
                    long_model.predict_proba(x_scaled)[:, 1],
                    dtype=float,
                )
                if long_model is not None
                else np.zeros(rows, dtype=float)
            ),
            "short": (
                np.asarray(
                    short_model.predict_proba(x_scaled)[:, 1],
                    dtype=float,
                )
                if short_model is not None
                else np.zeros(rows, dtype=float)
            ),
            "long_model_available": np.full(
                rows,
                long_model is not None,
                dtype=bool,
            ),
            "short_model_available": np.full(
                rows,
                short_model is not None,
                dtype=bool,
            ),
        }

    def predict_horizon_probabilities(self, x: np.ndarray) -> dict[str, float | np.ndarray]:
        x_scaled = self.scaler.transform(x)
        payload: dict[str, float | np.ndarray] = {"main": self.model.predict_proba(x_scaled)[:, 1]}
        for key, model in getattr(self, "auxiliary_models", {}).items():
            if str(key).startswith(
                ("direction_", "shadow_direction_", "alpha_")
            ):
                continue
            payload[key] = model.predict_proba(x_scaled)[:, 1]
        return payload

    def predict_expected_return(self, x: np.ndarray) -> np.ndarray:
        """Return continuous Alpha forecasts when an optional regressor exists."""

        x_scaled = self.scaler.transform(x)
        auxiliary = getattr(self, "auxiliary_models", {}) or {}
        expected_return_model = auxiliary.get("alpha_expected_return")
        if expected_return_model is not None and hasattr(
            expected_return_model,
            "predict_expected_return",
        ):
            values = expected_return_model.predict_expected_return(x_scaled)
            return np.asarray(values, dtype=float).reshape(-1)
        if hasattr(self.model, "predict_expected_return"):
            values = self.model.predict_expected_return(x_scaled)
            if values is not None:
                return np.asarray(values, dtype=float).reshape(-1)
        return np.full(len(x_scaled), np.nan, dtype=float)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as handle:
            pickle.dump(self, handle)
        return output

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        with Path(path).open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, ModelBundle):
            raise TypeError("Loaded object is not a ModelBundle")
        if not hasattr(bundle, "auxiliary_models"):
            bundle.auxiliary_models = {}
        if not hasattr(bundle, "auxiliary_metadata"):
            bundle.auxiliary_metadata = {}
        return bundle


def train_candidate_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_columns: list[str],
    seed: int = 42,
) -> ModelBundle:
    bundle, _ = train_candidate_models_with_report(x_train, y_train, x_valid, y_valid, feature_columns, seed)
    return bundle


def train_candidate_models_with_report(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_columns: list[str],
    seed: int = 42,
    enable_ensemble: bool = True,
) -> tuple[ModelBundle, dict[str, object]]:
    scaler = StandardScaler().fit(x_train)
    xs_train = scaler.transform(x_train)
    xs_valid = scaler.transform(x_valid)

    candidates, skipped = collect_candidate_models(seed)
    scored: list[tuple[float, ModelBundle, dict[str, object]]] = []
    for model in candidates:
        try:
            model.fit(xs_train, y_train)
            prob = model.predict_proba(xs_valid)[:, 1]
            metrics = classification_metrics(y_valid, prob)
            score = metrics["accuracy"] - metrics["log_loss"] * 0.05
            bundle = ModelBundle(
                model_name=model.name,
                model=model,
                scaler=scaler,
                feature_columns=feature_columns,
                metrics=metrics,
                config={"seed": seed},
            )
            entry = {
                "name": model.name,
                "status": "trained",
                "score": score,
                "metrics": metrics,
            }
            scored.append((score, bundle, entry))
        except Exception as exc:
            skipped.append({"name": getattr(model, "name", type(model).__name__), "reason": f"fit_failed: {exc}"})

    if enable_ensemble and len(scored) >= 2:
        top = sorted(scored, key=lambda item: item[0], reverse=True)[:4]
        ensemble = EnsembleProbabilityModel(
            name="ensemble_probability_average",
            models=[item[1].model for item in top],
            model_names=[item[1].model_name for item in top],
        )
        prob = ensemble.predict_proba(xs_valid)[:, 1]
        metrics = classification_metrics(y_valid, prob)
        score = metrics["accuracy"] - metrics["log_loss"] * 0.05 + 0.002
        bundle = ModelBundle(
            model_name=ensemble.name,
            model=ensemble,
            scaler=scaler,
            feature_columns=feature_columns,
            metrics=metrics,
            config={"seed": seed, "members": ",".join(ensemble.model_names)},
        )
        entry = {
            "name": ensemble.name,
            "status": "trained",
            "score": score,
            "metrics": metrics,
            "members": ensemble.model_names,
        }
        scored.append((score, bundle, entry))

    if not scored:
        raise RuntimeError(f"No candidate models trained successfully: {skipped}")

    scored.sort(key=lambda item: item[0], reverse=True)
    ranking = [item[2] for item in scored]
    report = {
        "ranking": ranking,
        "skipped": skipped,
        "best_model": scored[0][1].model_name,
    }
    scored[0][1].config = {**scored[0][1].config, "candidate_count": len(scored), "skipped_count": len(skipped)}
    return scored[0][1], report


def collect_candidate_models(seed: int = 42) -> tuple[list[object], list[dict[str, str]]]:
    candidates: list[object] = [
        LogisticRegressionNumpy(seed=seed),
        MLPClassifierNumpy(seed=seed),
    ]
    skipped: list[dict[str, str]] = []
    sklearn_candidates, sklearn_skipped = optional_sklearn_candidates_with_skips(seed)
    boosting_candidates, boosting_skipped = optional_boosting_candidates(seed)
    candidates.extend(sklearn_candidates)
    candidates.extend(boosting_candidates)
    skipped.extend(sklearn_skipped)
    skipped.extend(boosting_skipped)
    return candidates, skipped


def optional_sklearn_candidates(seed: int = 42) -> list[SklearnModelAdapter]:
    candidates, _ = optional_sklearn_candidates_with_skips(seed)
    return candidates


def optional_sklearn_candidates_with_skips(seed: int = 42) -> tuple[list[SklearnModelAdapter], list[dict[str, str]]]:
    if importlib.util.find_spec("sklearn") is None:
        return [], [{"name": "sklearn_models", "reason": "missing_dependency: sklearn"}]

    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    return [
        SklearnModelAdapter(
            "sklearn_hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=240,
                learning_rate=0.035,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed,
            ),
        ),
        SklearnModelAdapter(
            "sklearn_random_forest",
            RandomForestClassifier(
                n_estimators=240,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=1,
                random_state=seed,
                class_weight="balanced_subsample",
            ),
        ),
        SklearnModelAdapter(
            "sklearn_extra_trees",
            ExtraTreesClassifier(
                n_estimators=240,
                max_depth=8,
                min_samples_leaf=20,
                n_jobs=1,
                random_state=seed,
                class_weight="balanced",
            ),
        ),
        SklearnModelAdapter(
            "sklearn_logistic_regression",
            LogisticRegression(max_iter=1200, C=0.8, class_weight="balanced", random_state=seed),
        ),
        SklearnModelAdapter(
            "sklearn_mlp",
            MLPClassifier(
                hidden_layer_sizes=(48, 16),
                alpha=1e-4,
                learning_rate_init=0.001,
                max_iter=500,
                random_state=seed,
                early_stopping=True,
            ),
        ),
    ], []


def optional_boosting_candidates(seed: int = 42) -> tuple[list[object], list[dict[str, str]]]:
    candidates: list[object] = []
    skipped: list[dict[str, str]] = []

    if importlib.util.find_spec("lightgbm") is None:
        skipped.append({"name": "lightgbm_lgbm_classifier", "reason": "missing_dependency: lightgbm"})
    else:
        candidates.append(
            build_lightgbm_classifier_alpha(
                "lightgbm_lgbm_classifier",
                seed=seed,
                n_estimators=240,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
            )
        )

    if importlib.util.find_spec("xgboost") is None:
        skipped.append({"name": "xgboost_xgb_classifier", "reason": "missing_dependency: xgboost"})
    else:
        from xgboost import XGBClassifier

        candidates.append(
            SklearnModelAdapter(
                "xgboost_xgb_classifier",
                XGBClassifier(
                    n_estimators=240,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.0,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                    verbosity=0,
                ),
            )
        )

    if importlib.util.find_spec("catboost") is None:
        skipped.append({"name": "catboost_classifier", "reason": "missing_dependency: catboost"})
    else:
        from catboost import CatBoostClassifier

        candidates.append(
            SklearnModelAdapter(
                "catboost_classifier",
                CatBoostClassifier(
                    iterations=240,
                    depth=5,
                    learning_rate=0.035,
                    l2_leaf_reg=3.0,
                    loss_function="Logloss",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                ),
            )
        )

    return candidates, skipped


def optional_neural_candidates(seed: int = 42, complexity: str = "standard") -> tuple[list[TorchSequenceClassifierAdapter], list[dict[str, str]]]:
    if complexity != "blackbox":
        if complexity == "deep":
            return [], [{"name": "torch_sequence_transformer", "reason": "reserved_for_blackbox_complexity"}]
        return [], []
    if importlib.util.find_spec("torch") is None:
        return [], [{"name": "torch_sequence_transformer", "reason": "missing_dependency: torch"}]
    try:
        import torch

        torch.set_num_threads(1)
    except Exception as exc:
        return [], [{"name": "torch_sequence_transformer", "reason": f"import_failed: {exc}"}]

    return [
        TorchSequenceClassifierAdapter(
            name="torch_transformer_seq24_d32_l1_do015",
            sequence_length=24,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_feedforward=64,
            dropout=0.15,
            epochs=28,
            batch_size=128,
            learning_rate=8e-4,
            weight_decay=1e-4,
            patience=4,
            max_train_rows=9000,
            seed=seed,
        ),
        TorchSequenceClassifierAdapter(
            name="torch_transformer_seq48_d48_l1_do020",
            sequence_length=48,
            d_model=48,
            nhead=4,
            num_layers=1,
            dim_feedforward=96,
            dropout=0.20,
            epochs=24,
            batch_size=96,
            learning_rate=6e-4,
            weight_decay=2e-4,
            patience=4,
            max_train_rows=9000,
            seed=seed + 7,
        ),
    ], []
