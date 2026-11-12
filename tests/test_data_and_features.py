"""Unit tests: label alignment, splitting, scaling (leakage), windowing (causality)."""
import numpy as np
import pytest

from src.data.loader import label_vector, load_labels, select_channels
from src.data.validation import validate_dataset
from src.features.windows import forecast_windows, isolation_features
from src.preprocessing.scaling import ValueScaler, chronological_split_index


def test_label_vector_is_inclusive():
    assert label_vector(6, [[1, 2], [4, 4]]).tolist() == [0, 1, 1, 0, 1, 0]


def test_label_vector_rejects_out_of_range():
    with pytest.raises(ValueError):
        label_vector(5, [[3, 5]])


def test_load_labels_parses_lists(synthetic_raw_dir):
    labels = load_labels(synthetic_raw_dir)
    assert labels.loc[0, "anomaly_sequences"] == [[600, 639], [850, 850]]
    assert labels.loc[0, "classes"] == ["contextual", "point"]
    assert select_channels(labels, spacecraft="MSL") == ["M-1"]
    with pytest.raises(ValueError):
        select_channels(labels, channels=["NOPE"])


def test_validation_passes_on_clean_data(synthetic_raw_dir):
    report = validate_dataset(synthetic_raw_dir)
    assert report["ok"], report["errors"]
    assert report["summary"]["n_anomaly_sequences"] == 5


def test_validation_catches_length_mismatch(synthetic_raw_dir):
    csv = synthetic_raw_dir / "labeled_anomalies.csv"
    csv.write_text(csv.read_text().replace(",1000", ",999", 1))
    assert not validate_dataset(synthetic_raw_dir)["ok"]


def test_chronological_split():
    assert chronological_split_index(100, 0.2) == 80
    with pytest.raises(ValueError):
        chronological_split_index(100, 1.0)


def test_scaler_uses_only_fit_data():
    fit = np.array([[0.0], [2.0]], dtype=np.float32)
    test = np.array([[100.0]], dtype=np.float32)          # extreme test value must not change stats
    s = ValueScaler().fit(fit)
    assert (s.mean, s.std) == (1.0, 1.0)
    assert s.transform(test)[0, 0] == pytest.approx(99.0)
    assert ValueScaler.from_dict(s.to_dict()).mean == s.mean


def test_scaler_leaves_command_columns():
    arr = np.array([[1.0, 1.0], [3.0, 0.0]], dtype=np.float32)
    out = ValueScaler().fit(arr).transform(arr)
    assert out[:, 1].tolist() == [1.0, 0.0]


def test_forecast_windows_alignment():
    arr = np.arange(6, dtype=np.float32).reshape(-1, 1)  # values == timestep
    X, y, t = forecast_windows(arr, 2)
    assert t.tolist() == [2, 3, 4, 5]
    assert X[0, :, 0].tolist() == [0, 1] and y[0] == 2
    # every input timestep is strictly before its target (no look-ahead)
    assert (X[:, -1, 0] < y).all()


def test_isolation_features_are_causal():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((200, 3)).astype(np.float32)
    b = a.copy()
    b[150:, 0] += 50.0                                    # change the FUTURE only
    fa, names = isolation_features(a, lags=5, stat_window=20)
    fb, _ = isolation_features(b, lags=5, stat_window=20)
    np.testing.assert_array_equal(fa[:150], fb[:150])    # past features unchanged
    assert np.isnan(fa[:19]).any(axis=1).all() and np.isfinite(fa[19:]).all()
    assert "cmd_activity" in names


def test_merge_intervals_handles_benchmark_p2_duplicate():
    from src.data.loader import merge_intervals

    assert merge_intervals([[5350, 6575], [5300, 6420]], ["point", "point"]) == ([[5300, 6575]], ["point"])
    assert merge_intervals([[1, 2], [3, 4], [9, 9]], ["a", "b", "c"]) == ([[1, 4], [9, 9]], ["a", "c"])


def test_duplicate_label_rows_are_merged_and_reported(synthetic_raw_dir):
    csv = synthetic_raw_dir / "labeled_anomalies.csv"
    csv.write_text(csv.read_text() + 'S-1,SMAP,"[[595, 610]]",[contextual],1000\n')
    labels = load_labels(synthetic_raw_dir)
    assert len(labels) == 3
    assert labels.set_index("chan_id").loc["S-1", "anomaly_sequences"] == [[595, 639], [850, 850]]
    report = validate_dataset(synthetic_raw_dir)
    assert report["ok"] and any("merged" in w for w in report["warnings"])


def test_scaler_constant_channel_keeps_units():
    s = ValueScaler().fit(np.full((50, 1), -1.0, dtype=np.float32))
    assert s.std == 1.0
    assert s.transform(np.array([[0.0]], dtype=np.float32))[0, 0] == pytest.approx(1.0)
