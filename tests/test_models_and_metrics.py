"""Unit tests: detectors, thresholds, metrics, persistence and inference."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import event_counts, event_metrics, macro_auc, point_counts, point_metrics, threshold_free
from src.inference.predict import detect, frame_to_array
from src.models.detectors import IsolationForestDetector, RollingZScoreDetector
from src.models.lstm import LSTMDetector, LSTMForecaster
from src.models.persistence import DetectorBundle, load_bundle
from src.models.thresholds import apply_threshold, calibrate_threshold, calibration_is_degenerate, runs
from src.preprocessing.scaling import ValueScaler


def _signal(n=600, seed=0):
    rng = np.random.default_rng(seed)
    v = np.sin(np.arange(n) / 8) + 0.05 * rng.standard_normal(n)
    return np.column_stack([v, np.zeros(n)]).astype(np.float32)


# ---------- metrics ----------
def test_point_metrics_hand_example():
    c = point_counts(np.array([1, 1, 0, 0, 0]), np.array([1, 0, 1, 0, 0]))
    assert c == {"tp": 1, "fp": 1, "tn": 2, "fn": 1}
    m = point_metrics(c)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert m["fpr"] == pytest.approx(1 / 3) and m["accuracy"] == 0.6


def test_accuracy_misleads_on_imbalance():
    y = np.zeros(1000, dtype=int); y[:10] = 1
    m = point_metrics(point_counts(y, np.zeros(1000, dtype=int)))
    assert m["accuracy"] == 0.99 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_event_metrics():
    pred = np.zeros(30, dtype=int)
    pred[[3, 4]] = 1          # hits event [2,6] at offset 1
    pred[20:23] = 1           # false alarm run
    c = event_counts([[2, 6], [10, 12]], pred)
    assert (c["event_tp"], c["event_fn"], c["event_fp"]) == (1, 1, 1)
    assert c["latencies"] == [1]
    assert event_metrics(c)["event_f1"] == pytest.approx(0.5)


def test_threshold_free_auc_perfect_ranking():
    r = threshold_free(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 2.0, 3.0]))
    assert r["roc_auc"] == 1.0 and r["pr_auc"] == 1.0


def test_threshold_free_ignores_warmup_nans():
    y, s = np.array([0, 0, 1, 1, 0]), np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
    assert threshold_free(y, s)["roc_auc"] == 0.0            # only the 3 finite points count (0.667 if NaN -> 0)
    assert threshold_free(np.array([0, 0]), np.array([1.0, 2.0]))["roc_auc"] is None


def test_macro_auc_mean_and_bootstrap_interval():
    r = macro_auc([0.5, 0.7, 0.9, None])
    assert r["n"] == 3 and r["mean"] == pytest.approx(0.7)
    assert r["ci_low"] <= r["mean"] <= r["ci_high"] and r["ci_low"] >= 0.5 and r["ci_high"] <= 0.9
    assert macro_auc([])["mean"] is None


def test_degenerate_calibration_is_flagged():
    zeros = np.zeros(500)
    assert calibration_is_degenerate(zeros, calibrate_threshold(zeros))         # constant channel
    assert calibration_is_degenerate(np.full(500, 0.2), 0.2)                      # no spread at all
    s = np.abs(np.random.default_rng(0).standard_normal(500))
    assert not calibration_is_degenerate(s, calibrate_threshold(s))


# ---------- thresholds ----------
def test_calibrate_threshold_ignores_nan_and_uses_quantile():
    s = np.concatenate([[np.nan] * 5, np.arange(101, dtype=float)])
    assert calibrate_threshold(s, "quantile", quantile=0.9) == pytest.approx(90.0)
    assert calibrate_threshold(np.zeros(10)) > 0           # never a zero threshold
    with pytest.raises(ValueError):
        calibrate_threshold(np.array([np.nan]))


def test_runs_and_postprocessing():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 2), (4, 4)]
    s = np.array([np.nan, 5, 0, 5, 0, 0, 5])
    assert apply_threshold(s, 1).tolist() == [0, 1, 0, 1, 0, 0, 1]
    assert apply_threshold(s, 1, merge_gap=1).tolist() == [0, 1, 1, 1, 0, 0, 1]
    assert apply_threshold(s, 1, min_event_length=2, merge_gap=1).tolist() == [0, 1, 1, 1, 0, 0, 0]


# ---------- detectors ----------
def test_baseline_is_causal_and_detects_spike():
    a = _signal()
    a[400, 0] = 8.0
    s = RollingZScoreDetector(window=50).score(a)
    assert np.isnan(s[:50]).all()
    assert np.nanargmax(s) == 400
    b = a.copy(); b[401:, 0] = 100.0                      # future change must not alter score at 400
    assert RollingZScoreDetector(window=50).score(b)[400] == s[400]


def test_isolation_forest_ranks_spike_high():
    a = _signal()
    det = IsolationForestDetector(lags=5, stat_window=20, n_estimators=50).fit(a, 400)
    b = a.copy(); b[500, 0] = 6.0
    s = det.score(b)
    assert s[500] > np.nanquantile(s, 0.99)


def test_lstm_forward_shape_and_learning():
    torch.manual_seed(0)
    model = LSTMForecaster(n_features=3, hidden_size=8, num_layers=2)
    assert model(torch.zeros(4, 10, 3)).shape == (4,)
    det = LSTMDetector(window=20, hidden_size=16, num_layers=1, epochs=8, patience=8, max_train_windows=400)
    det.fit(_signal(), 480)
    assert det.history["val_loss"][det.best_epoch] < det.history["val_loss"][0]
    pred = det.predict(_signal())
    assert np.isnan(pred[:20]).all() and np.isfinite(pred[20:]).all()


# ---------- persistence + inference ----------
@pytest.mark.parametrize("method", ["baseline", "isolation_forest", "lstm"])
def test_bundle_roundtrip_gives_identical_scores(tmp_path, method):
    raw = _signal()
    scaler = ValueScaler().fit(raw[:480])
    det = {"baseline": RollingZScoreDetector(window=30),
           "isolation_forest": IsolationForestDetector(lags=5, stat_window=20, n_estimators=30),
           "lstm": LSTMDetector(window=20, hidden_size=8, num_layers=1, epochs=2)}[method]
    det.fit(scaler.transform(raw), 480)
    bundle = DetectorBundle(method, "X-1", det, scaler, 1.23, raw.shape[1], {"min_event_length": 1, "merge_gap": 0})
    bundle.save(tmp_path)
    loaded = load_bundle(tmp_path, method, "X-1")
    assert loaded.threshold == 1.23
    np.testing.assert_allclose(detect(loaded, raw)["score"], detect(bundle, raw)["score"], rtol=1e-5, equal_nan=True)


def test_frame_to_array_validation():
    df = pd.DataFrame({"value": [0.1, 0.2, 0.3]})
    arr, warns = frame_to_array(df, n_features=3)
    assert arr.shape == (3, 3) and warns                   # warns that commands assumed 0
    with pytest.raises(ValueError):
        frame_to_array(pd.DataFrame({"x": [1]}), 1)
    with pytest.raises(ValueError):
        frame_to_array(pd.DataFrame({"value": [1.0, np.nan]}), 1)
    with pytest.raises(ValueError):
        frame_to_array(pd.DataFrame({"value": [0.1], "cmd_0": [2]}), 2)


def test_detect_rejects_too_short_input(tmp_path):
    bundle = DetectorBundle("baseline", "X-1", RollingZScoreDetector(window=50), ValueScaler(), 1.0, 2, {})
    with pytest.raises(ValueError):
        detect(bundle, _signal(30))


@pytest.mark.parametrize("method", ["baseline", "isolation_forest", "lstm"])
def test_validation_scores_use_test_like_conditions(method):
    """Validation scores must not depend on fit-part data older than the detector's warm-up
    (e.g. LSTM EWMA must start fresh at the split, exactly like it does on the test file)."""
    from src.pipeline import validation_scores

    a = _signal(600)
    det = {"baseline": RollingZScoreDetector(window=30),
           "isolation_forest": IsolationForestDetector(lags=5, stat_window=20, n_estimators=30),
           "lstm": LSTMDetector(window=20, hidden_size=8, num_layers=1, epochs=2)}[method].fit(a, 480)
    v1 = validation_scores(det, a, 480)
    b = a.copy()
    b[: 480 - det.warmup, 0] += 5.0               # perturb old fit-part data only
    v2 = validation_scores(det, b, 480)
    assert len(v1) == 120 and np.isfinite(v1).all()
    np.testing.assert_allclose(v1, v2)
