"""LSTM one-step-ahead forecaster used as an anomaly detector (Telemanom-style methodology).

Idea: learn what *normal* telemetry looks like by predicting the next value from the last W
timesteps (value + command flags). Where the model is surprised - large |actual - predicted| -
the behaviour differs from what it learned during nominal operation.

    e_t     = |x_t - x̂_t|                          (prediction error)
    score_t = EWMA_span(e)_t                        (smoothed, causal: uses e_0..e_t only)

Why forecasting rather than a reconstruction autoencoder? The benchmark's labels mark *when*
behaviour departs from expectation; forecasting error is aligned to a single timestep t, which
keeps the label alignment simple and matches the original paper. An LSTM (rather than an MLP over
lags) carries a learned memory (cell state) across the window, so it can model regimes that depend
on commands issued many steps earlier.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.loader import VALUE_COL
from src.features.windows import forecast_windows
from src.utils.reproducibility import get_logger

log = get_logger("lstm")


class InsufficientDataError(ValueError):
    """The fit part of a channel is too short to form a single training window (e.g. D-12 with l_s = 250)."""


class LSTMForecaster(nn.Module):
    """Stacked LSTM -> linear head that predicts the next (scaled) telemetry value."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3,
                 n_outputs: int = 1):
        super().__init__()
        self.n_outputs = int(n_outputs)
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, self.n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, window, n_features)
        out, _ = self.lstm(x)               # out: (batch, window, hidden) - hidden state at every step
        last = out[:, -1, :]                # summary of the whole window = hidden state at the last step
        y = self.head(self.dropout(last))   # (batch, n_outputs): forecasts for t, t+1, ..., t+n_outputs-1
        return y.squeeze(-1) if self.n_outputs == 1 else y


def resolve_device(device: str = "cpu") -> str:
    """'auto' -> 'cuda' when a CUDA GPU is usable, else 'cpu'. An explicit 'cuda' without a GPU falls back to CPU
    with a warning instead of crashing. MPS (Apple) is used only when asked for explicitly: it is not verified here."""
    device = str(device or "cpu").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("device=%s requested but CUDA is not available - using CPU", device)
        return "cpu"
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        log.warning("device=mps requested but MPS is not available - using CPU")
        return "cpu"
    return device


class LSTMDetector:
    name = "lstm"

    def __init__(self, window: int = 100, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3,
                 batch_size: int = 64, epochs: int = 30, learning_rate: float = 1e-3, patience: int = 5,
                 max_train_windows: int = 4000, use_commands: bool = True, smoothing_span: int = 30,
                 device: str = "cpu", seed: int = 42, n_predictions: int = 1, min_delta: float = 1e-6,
                 restore_best: bool = True, warmup_fill: int = 0, **_ignored):
        self.window, self.batch_size, self.epochs = int(window), int(batch_size), int(epochs)
        self.hidden_size, self.num_layers, self.dropout = int(hidden_size), int(num_layers), float(dropout)
        self.learning_rate, self.patience = float(learning_rate), int(patience)
        self.max_train_windows, self.use_commands = int(max_train_windows), bool(use_commands)
        self.smoothing_span, self.seed = int(smoothing_span), int(seed)
        self.device_requested = device
        self.train_device: str | None = None
        self.device = resolve_device(device)
        # Telemanom-style options (defaults reproduce the original LSTM detector exactly):
        #   n_predictions  forecast horizon trained jointly; only the 1-step forecast is scored
        #   min_delta      improvement needed to reset early-stopping patience
        #   restore_best   keep the best-validation epoch (True) or the last epoch (Keras default, False)
        #   warmup_fill    replace the first k smoothed errors with the mean of the first 2k (Telemanom), 0 = off
        self.n_predictions, self.min_delta = int(n_predictions), float(min_delta)
        self.restore_best, self.warmup_fill = bool(restore_best), int(warmup_fill)
        self.model: LSTMForecaster | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self.best_epoch: int | None = None

    # -- helpers -----------------------------------------------------------------------------
    def _inputs(self, array: np.ndarray) -> np.ndarray:
        return array if self.use_commands else array[:, [VALUE_COL]]

    def _predict_windows(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "fit() or load first"
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.from_numpy(np.ascontiguousarray(X[i : i + 1024])).to(self.device)
                out = self.model(xb)
                preds.append((out if out.ndim == 1 else out[:, 0]).cpu().numpy())   # 1-step-ahead forecast
        return np.concatenate(preds) if preds else np.empty(0)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.n_predictions == 1:      # original code path, kept bit-for-bit for the reference runs
            return float(np.mean((self._predict_windows(X) - y) ** 2))
        assert self.model is not None
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.from_numpy(np.ascontiguousarray(X[i : i + 1024])).to(self.device)
                yb = torch.from_numpy(np.ascontiguousarray(y[i : i + 1024])).to(self.device)
                total += float(((self.model(xb) - yb) ** 2).sum().item())
        return total / y.size

    def _targets(self, values: np.ndarray, t_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Targets for windows ending before t: value[t .. t+k-1]. Windows whose horizon runs past the end are dropped."""
        k = self.n_predictions
        ok = t_idx + k - 1 < len(values)
        if k == 1:
            return values[t_idx].astype(np.float32), ok
        Y = np.stack([values[np.minimum(t_idx + j, len(values) - 1)] for j in range(k)], axis=1).astype(np.float32)
        return Y, ok

    # -- public API --------------------------------------------------------------------------
    def fit(self, train_scaled: np.ndarray, split_idx: int) -> "LSTMDetector":
        arr = self._inputs(train_scaled)
        X, _, t_idx = forecast_windows(arr, self.window)
        y, ok = self._targets(arr[:, VALUE_COL], t_idx)
        # A window belongs to training only if ALL its TARGETS are before the split; its inputs are then
        # also before the split. Validation windows (target >= split) may look back into the fit
        # region - that is past data, which is legitimate - but are never trained on.
        last_target = t_idx + self.n_predictions - 1
        tr, va = ok & (last_target < split_idx), ok & (t_idx >= split_idx)
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
        if len(Xtr) == 0 or len(Xva) == 0:
            raise InsufficientDataError(f"Not enough data for LSTM train/validation windows (window {self.window}, "
                                        f"horizon {self.n_predictions}, fit part {split_idx})")
        if 0 < self.max_train_windows < len(Xtr):  # even stride (0 = no cap) keeps coverage of the whole period
            keep = np.linspace(0, len(Xtr) - 1, self.max_train_windows).round().astype(int)
            Xtr, ytr = Xtr[keep], ytr[keep]

        # Seed twice: before construction (weight init) and again before training (dropout masks). The detector's
        # own seed then fully determines the result, independent of the caller's global RNG state, and runs made
        # before this fix (where the pipeline's set_seed() supplied the first seeding) are reproduced exactly.
        torch.manual_seed(self.seed)
        self.model = LSTMForecaster(arr.shape[1], self.hidden_size, self.num_layers, self.dropout,
                                    self.n_predictions).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)    # window shuffling order
        best_val, best_state, bad_epochs = np.inf, None, 0
        for epoch in range(self.epochs):
            self.model.train()
            order = torch.randperm(len(Xtr), generator=gen).numpy()  # shuffle WINDOWS, not timesteps
            total = 0.0
            for i in range(0, len(order), self.batch_size):
                b = order[i : i + self.batch_size]
                xb = torch.from_numpy(np.ascontiguousarray(Xtr[b])).to(self.device)
                yb = torch.from_numpy(ytr[b]).to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # guards against exploding gradients
                opt.step()
                total += loss.item() * len(b)
            val_loss = self._loss(Xva, yva)
            if not np.isfinite(val_loss):
                raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
            self.history["train_loss"].append(total / len(order))
            self.history["val_loss"].append(val_loss)
            if val_loss < best_val - self.min_delta:
                best_val, best_state, bad_epochs, self.best_epoch = val_loss, copy.deepcopy(self.model.state_dict()), 0, epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is None:  # only possible if epochs == 0
            raise RuntimeError("LSTM training produced no epoch to keep")
        if self.restore_best:
            self.model.load_state_dict(best_state)  # early stopping: keep the best validation epoch
        else:
            self.best_epoch = len(self.history["val_loss"]) - 1   # Keras EarlyStopping default: keep the last epoch
        # Evidence of where training really ran: the parameters' device during the last epoch (not the request).
        self.train_device = str(next(self.model.parameters()).device)
        if self.train_device.startswith("cuda"):
            self.train_device += f" ({torch.cuda.get_device_name(next(self.model.parameters()).device)})"
        self.model.to("cpu")
        self.device = "cpu"                 # scoring/inference always on CPU: identical numbers wherever it was trained
        return self

    def predict(self, array_scaled: np.ndarray) -> np.ndarray:
        """One-step-ahead predictions aligned to timesteps (NaN for the first ``window`` steps)."""
        X, _, t_idx = forecast_windows(self._inputs(array_scaled), self.window)
        pred = np.full(len(array_scaled), np.nan)
        pred[t_idx] = self._predict_windows(X)
        return pred

    def score(self, array_scaled: np.ndarray) -> np.ndarray:
        err = np.abs(array_scaled[:, VALUE_COL] - self.predict(array_scaled))
        return smooth_errors(err, self.smoothing_span, self.warmup_fill)

    @property
    def warmup(self) -> int:
        return self.window

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in ("window", "hidden_size", "num_layers", "dropout", "batch_size",
                                             "epochs", "learning_rate", "patience", "max_train_windows",
                                             "use_commands", "smoothing_span", "seed", "n_predictions",
                                             "min_delta", "restore_best", "warmup_fill")}


def smooth_errors(err: np.ndarray, span: int, warmup_fill: int = 0) -> np.ndarray:
    """Causal EWMA of absolute errors (NaN warm-up positions stay NaN).

    ``warmup_fill = k > 0`` reproduces Telemanom's start-up correction: the first k smoothed errors are replaced by
    the mean of the first 2k. That uses up to 2k errors ahead, but Telemanom's dynamic threshold only decides those
    positions once its first 2,100-error window is complete, so it does not leak into an earlier decision.
    """
    err = np.asarray(err, dtype=np.float64)
    e_s = np.array(pd.Series(err).ewm(span=span, ignore_na=True).mean().where(~np.isnan(err)).to_numpy())
    if warmup_fill > 0:
        ok = np.flatnonzero(np.isfinite(e_s))
        if ok.size:
            head = ok[: warmup_fill]
            e_s[head] = np.mean(e_s[ok[: 2 * warmup_fill]])
    return e_s


class TelemanomDetector(LSTMDetector):
    """Telemanom-aligned forecaster (Hundman et al., 2018): same LSTM family, paper hyper-parameters
    (l_s = 250 inputs, 2 x 80 units, 10-step horizon, EWMA span 105, Keras-style early stopping), and its
    predictions come from the nonparametric dynamic threshold (``src.models.dynamic_threshold``), not from a
    fixed validation quantile. Not an exact reproduction - see docs/methodology.md for the differences."""

    name = "telemanom"
    threshold_mode = "dynamic"
