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


def test_lstm_seed_alone_controls_weights():
    """Regression (review D5): the detector's own seed must fix the weights, whatever the global RNG state."""
    weights = []
    for prior in (0, 1):
        torch.manual_seed(prior)
        det = LSTMDetector(window=20, hidden_size=8, num_layers=1, epochs=1, seed=42).fit(_signal(), 480)
        weights.append({k: v.clone() for k, v in det.model.state_dict().items()})
    assert all(torch.equal(weights[0][k], weights[1][k]) for k in weights[0])


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
    assert loaded.threshold == 1.23 and loaded.calibratable is None and loaded.calibration_warning()
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


@pytest.mark.parametrize("bad", ["../lstm/P-1", "P-1/../../x", "/etc/passwd", "p-1", "", "P-1\n"])
def test_load_bundle_rejects_path_like_channel_ids(tmp_path, bad):
    """Regression (review D7): a channel id must never be able to point outside the models directory."""
    with pytest.raises(ValueError):
        load_bundle(tmp_path, "lstm", bad)


def test_read_telemetry_csv_bounds_rows(tmp_path, monkeypatch):
    from src.inference import predict

    monkeypatch.setattr(predict, "MAX_ROWS", 5)
    f = tmp_path / "big.csv"
    f.write_text("value\n" + "0.1\n" * 6)
    with pytest.raises(ValueError):
        predict.read_telemetry_csv(f)
    f.write_text("value\n" + "0.1\n" * 5)
    assert len(predict.read_telemetry_csv(f)) == 5


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


# ---------- calibration studies (review D4 + leave-channels-out) ----------
def _records(n_chan=10, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_chan):
        y = np.zeros(400, dtype=np.int8); y[200:220] = 1
        test = rng.random(400); test[200:220] += 1.0
        recs.append({"chan_id": f"X-{i}", "spacecraft": "SMAP" if i % 3 else "MSL", "y": y, "seqs": [[200, 219]],
                     "calibratable": True, "threshold": {"lstm": float(np.quantile(v := rng.random(150), 0.995))},
                     "test": {"lstm": test}, "val": {"lstm": v}})
    return recs


def test_rule_threshold_default_reproduces_calibrated_threshold():
    from src.evaluation.calibration import rule_threshold

    v = np.abs(np.random.default_rng(1).standard_normal(300))
    assert rule_threshold(v, 0.995, 1.0) == calibrate_threshold(v, "quantile", quantile=0.995)   # exactly (audit A1)
    assert rule_threshold(np.zeros(50), 0.995, 3.0) > 0                       # same eps floor


def test_lco_folds_partition_and_stratify():
    from src.evaluation.calibration import make_folds

    recs = _records(12)
    folds = make_folds(recs, k=3)
    assert sorted(i for f in folds for i in f) == list(range(12))
    assert all(sum(recs[i]["spacecraft"] == "MSL" for i in f) >= 1 for f in folds)


def test_lco_rule_ignores_heldout_labels():
    """The rule applied to a held-out fold must not change when that fold's labels change."""
    from src.evaluation.calibration import leave_channels_out, make_folds

    recs = _records(10)
    base = leave_channels_out(recs, ["lstm"], k=5)
    held = make_folds(recs, k=5)[0]
    for i in held:                                            # scramble fold 0's labels only
        recs[i]["y"] = np.roll(recs[i]["y"], 100); recs[i]["seqs"] = [[300, 319]]
    again = leave_channels_out(recs, ["lstm"], k=5)
    assert again["lstm"]["chosen_rules"][0] == base["lstm"]["chosen_rules"][0]


def test_calibration_uncertainty_shapes_and_order_independence():
    """Intervals are well formed, and (regression, final audit A3) one method's interval must not depend on
    which other methods are evaluated in the same call."""
    from src.evaluation.calibration import calibration_uncertainty

    recs = _records(6)
    for r in recs:                                          # a second "method" with different scores
        r["val"]["other"], r["test"]["other"], r["threshold"]["other"] = r["val"]["lstm"] * 2, r["test"]["lstm"] * 2, 1.0
    alone = calibration_uncertainty(recs, ["lstm"], n_boot=30)["lstm"]
    together = calibration_uncertainty(recs, ["other", "lstm"], n_boot=30)["lstm"]
    assert alone["interval_90"] == together["interval_90"]
    assert len(alone["per_channel"]) == 6 and alone["per_channel"][0]["expected_exceedances"] == pytest.approx(0.75)
    lo, hi = alone["interval_90"]["f1"]
    assert 0.0 <= lo <= hi <= 1.0


def test_block_bootstrap_resamples_contiguous_blocks():
    from src.evaluation.calibration import block_bootstrap_indices

    idx = block_bootstrap_indices(100, 5, 10, np.random.default_rng(0))
    assert idx.shape == (5, 100)
    steps = np.diff(idx, axis=1) % 100
    assert (steps[:, :9] == 1).all()                        # first block is a contiguous (circular) run
    assert block_bootstrap_indices(7, 3, 50, np.random.default_rng(0)).shape == (3, 7)   # block longer than series


def test_bundle_records_calibration_status(tmp_path):
    """An uncalibratable model must say so at inference time (constant training channel -> threshold ~0)."""
    det = RollingZScoreDetector(window=30)
    DetectorBundle("baseline", "X-1", det, ValueScaler(), 1e-7, 2, {}, calibratable=False).save(tmp_path)
    loaded = load_bundle(tmp_path, "baseline", "X-1")
    assert loaded.calibratable is False and "NOT calibrated" in loaded.calibration_warning()
    DetectorBundle("baseline", "X-2", det, ValueScaler(), 1.0, 2, {}, calibratable=True).save(tmp_path)
    assert load_bundle(tmp_path, "baseline", "X-2").calibration_warning() is None
