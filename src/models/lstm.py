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


class LSTMForecaster(nn.Module):
    """Stacked LSTM -> linear head that predicts the next (scaled) telemetry value."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, window, n_features)
        out, _ = self.lstm(x)               # out: (batch, window, hidden) - hidden state at every step
        last = out[:, -1, :]                # summary of the whole window = hidden state at the last step
        return self.head(self.dropout(last)).squeeze(-1)  # (batch,)


class LSTMDetector:
    name = "lstm"

    def __init__(self, window: int = 100, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3,
                 batch_size: int = 64, epochs: int = 30, learning_rate: float = 1e-3, patience: int = 5,
                 max_train_windows: int = 4000, use_commands: bool = True, smoothing_span: int = 30,
                 device: str = "cpu", seed: int = 42, **_ignored):
        self.window, self.batch_size, self.epochs = int(window), int(batch_size), int(epochs)
        self.hidden_size, self.num_layers, self.dropout = int(hidden_size), int(num_layers), float(dropout)
        self.learning_rate, self.patience = float(learning_rate), int(patience)
        self.max_train_windows, self.use_commands = int(max_train_windows), bool(use_commands)
        self.smoothing_span, self.device, self.seed = int(smoothing_span), device, int(seed)
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
                preds.append(self.model(xb).cpu().numpy())
        return np.concatenate(preds) if preds else np.empty(0)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self._predict_windows(X) - y) ** 2))

    # -- public API --------------------------------------------------------------------------
    def fit(self, train_scaled: np.ndarray, split_idx: int) -> "LSTMDetector":
        arr = self._inputs(train_scaled)
        X, y, t_idx = forecast_windows(arr, self.window)
        # A window belongs to training only if its TARGET is before the split; its inputs are then
        # also before the split. Validation windows (target >= split) may look back into the fit
        # region - that is past data, which is legitimate - but are never trained on.
        tr, va = t_idx < split_idx, t_idx >= split_idx
        Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
        if len(Xtr) == 0 or len(Xva) == 0:
            raise ValueError("Not enough data for LSTM train/validation windows")
        if len(Xtr) > self.max_train_windows:  # even stride keeps coverage of the whole period
            keep = np.linspace(0, len(Xtr) - 1, self.max_train_windows).round().astype(int)
            Xtr, ytr = Xtr[keep], ytr[keep]

        # Seed twice: before construction (weight init) and again before training (dropout masks). The detector's
        # own seed then fully determines the result, independent of the caller's global RNG state, and runs made
        # before this fix (where the pipeline's set_seed() supplied the first seeding) are reproduced exactly.
        torch.manual_seed(self.seed)
        self.model = LSTMForecaster(arr.shape[1], self.hidden_size, self.num_layers, self.dropout).to(self.device)
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
            if val_loss < best_val - 1e-6:
                best_val, best_state, bad_epochs, self.best_epoch = val_loss, copy.deepcopy(self.model.state_dict()), 0, epoch
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is None:  # only possible if epochs == 0
            raise RuntimeError("LSTM training produced no epoch to keep")
        self.model.load_state_dict(best_state)  # early stopping: keep the best validation epoch
        return self

    def predict(self, array_scaled: np.ndarray) -> np.ndarray:
        """One-step-ahead predictions aligned to timesteps (NaN for the first ``window`` steps)."""
        X, _, t_idx = forecast_windows(self._inputs(array_scaled), self.window)
        pred = np.full(len(array_scaled), np.nan)
        pred[t_idx] = self._predict_windows(X)
        return pred

    def score(self, array_scaled: np.ndarray) -> np.ndarray:
        err = np.abs(array_scaled[:, VALUE_COL] - self.predict(array_scaled))
        # pandas ewm is causal (each output uses only current and past errors). NaNs stay NaN.
        return pd.Series(err).ewm(span=self.smoothing_span, ignore_na=True).mean().where(~np.isnan(err)).to_numpy()

    @property
    def warmup(self) -> int:
        return self.window

    def get_params(self) -> dict:
        return {k: getattr(self, k) for k in ("window", "hidden_size", "num_layers", "dropout", "batch_size",
                                             "epochs", "learning_rate", "patience", "max_train_windows",
                                             "use_commands", "smoothing_span", "seed")}
