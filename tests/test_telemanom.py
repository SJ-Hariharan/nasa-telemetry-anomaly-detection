"""Telemanom-aligned forecaster, dynamic threshold, derived (lstm_dynamic) runs, sharding and device selection."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models.dynamic_threshold import (DynamicThresholdConfig, dynamic_threshold, find_epsilon, last_decided_index,
                                          release_times, window_bounds)
from src.models.thresholds import runs

CFG = DynamicThresholdConfig()


def _errors(n=6000, seed=0, spikes=((3000, 3060),), height=1.5):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    values = 0.6 * np.sin(2 * np.pi * t / 300)
    e = np.abs(0.03 * rng.standard_normal(n))
    for s, e_ in spikes:
        e[s:e_] += height
    e_s = pd.Series(e).ewm(span=105).mean().to_numpy()
    return e_s, values


# ---------- dynamic threshold ----------
def test_windows_cover_every_position_once():
    for n in (500, 2100, 2101, 2169, 2170, 7777):
        decided = np.zeros(n, int)
        for start, end, dec in window_bounds(n, CFG):
            decided[start + dec : end] += 1
        ignore = 2 * CFG.l_s if n >= 2500 else (CFG.l_s if n >= 1800 else 0)
        assert (decided[ignore:] == 1).all(), n      # no gap (the reference code leaves one) and no double decision


def test_clear_error_burst_is_flagged_and_noise_is_not():
    e_s, v = _errors()
    pred, eps = dynamic_threshold(e_s, v)
    seqs = runs(pred)
    assert seqs and all(s <= 3060 and e >= 3000 for s, e in seqs)   # only around the burst
    assert np.isfinite(eps[500:]).all()
    quiet, v2 = _errors(spikes=())
    assert dynamic_threshold(quiet, v2)[0].sum() == 0                 # pruning rejects pure noise


def test_small_errors_relative_to_signal_are_skipped():
    """The 5 % rules: an error burst that is tiny compared with the telemetry range is not an anomaly."""
    e_s, v = _errors(height=0.02)
    assert dynamic_threshold(e_s, v * 50)[0].sum() == 0


def test_leading_nan_warmup_is_respected_and_aligned():
    e_s, v = _errors()
    e2 = np.concatenate([np.full(250, np.nan), e_s])
    v2 = np.concatenate([np.zeros(250), v])
    p1, _ = dynamic_threshold(e_s, v)
    p2, eps2 = dynamic_threshold(e2, v2)
    assert (p2[:250] == 0).all() and np.isnan(eps2[:250]).all()
    np.testing.assert_array_equal(p1, p2[250:])
    with pytest.raises(ValueError):
        dynamic_threshold(np.array([0.1, np.nan, 0.2]), np.zeros(3))


def test_replay_prefix_matches_final_decisions():
    """What the replay shows at time t is exactly the final decision for every position already released."""
    e_s, v = _errors(n=5000, spikes=((2500, 2560), (4200, 4230)))
    final, _ = dynamic_threshold(e_s, v)
    rel = release_times(e_s)
    for t in (2100, 2400, 2590, 3333, 4300):
        part, _ = dynamic_threshold(e_s, v, decided_until=t)
        done = rel < t
        # groups that straddle the release boundary may still grow; everything strictly decided must agree
        stable = done & (np.arange(len(e_s)) < last_decided_index(t) - CFG.error_buffer)
        np.testing.assert_array_equal(part[stable], final[stable])
        assert part[~done].sum() == 0                      # nothing is shown before it is decided


def test_release_times_are_causal_and_bounded():
    e_s = np.concatenate([np.full(100, np.nan), _errors(n=4000)[0]])
    rel = release_times(e_s)
    assert (rel >= np.arange(len(e_s))).all()
    after_first = np.arange(100 + 2100, len(e_s))
    assert (rel[after_first] - after_first).max() < CFG.batch_size     # at most one batch of delay


def test_find_epsilon_defaults_when_flat():
    eps, z = find_epsilon(np.full(300, 0.2), CFG)
    assert z == CFG.sd_lim and eps == pytest.approx(0.2)


def test_pruning_parameter_trades_recall_for_false_alarms():
    e_s, v = _errors(spikes=((3000, 3060), (4500, 4510)), height=0.4)
    n_strict = dynamic_threshold(e_s, v, {"p": 0.9})[0].sum()
    n_loose = dynamic_threshold(e_s, v, {"p": 0.01})[0].sum()
    assert n_strict <= n_loose


# ---------- forecaster ----------
def _signal(n=900, n_cmd=3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    v = 0.5 * np.sin(2 * np.pi * t / 40) + 0.02 * rng.standard_normal(n)
    return np.column_stack([v, (rng.random((n, n_cmd)) < 0.02)]).astype(np.float32)


def test_multistep_forecaster_trains_and_scores_one_step():
    from src.models.lstm import TelemanomDetector

    a = _signal()
    det = TelemanomDetector(window=30, hidden_size=8, num_layers=2, epochs=2, n_predictions=5, restore_best=False,
                            warmup_fill=20, smoothing_span=10, max_train_windows=0).fit(a, 700)
    s = det.score(a)
    assert np.isnan(s[:30]).all() and np.isfinite(s[30:]).all()
    assert np.allclose(s[30:50], s[30:50].mean())          # warm-up fill: first k smoothed errors are constant
    assert det.best_epoch == len(det.history["val_loss"]) - 1   # Keras-style: last epoch kept


def test_telemanom_training_windows_never_see_validation_targets():
    from src.models.lstm import LSTMDetector

    det = LSTMDetector(window=10, n_predictions=4)
    t_idx = np.arange(10, 100)
    _, ok = det._targets(np.arange(100, dtype=np.float32), t_idx)
    assert ok.sum() == 87                                   # horizon must stay inside the series
    last = t_idx + det.n_predictions - 1
    assert (last[ok & (last < 60)] < 60).all()


def test_telemanom_bundle_roundtrip_and_dynamic_inference(tmp_path):
    from src.inference.predict import detect
    from src.models.lstm import TelemanomDetector
    from src.models.persistence import DetectorBundle, load_bundle
    from src.preprocessing.scaling import ValueScaler

    a = _signal(2600)
    a[2300:2340, 0] += 1.5
    det = TelemanomDetector(window=30, hidden_size=8, num_layers=1, epochs=2, n_predictions=3,
                            smoothing_span=20, warmup_fill=30).fit(a, 1800)
    b = DetectorBundle("telemanom", "X-1", det, ValueScaler(0, 1), 0.1, a.shape[1], {}, True, {"p": 0.13})
    b.save(tmp_path)
    loaded = load_bundle(tmp_path, "telemanom", "X-1")
    assert loaded.dynamic == {"p": 0.13} and loaded.detector.n_predictions == 3
    out1, out2 = detect(b, a), detect(loaded, a)
    np.testing.assert_allclose(out1["score"], out2["score"], rtol=1e-5, equal_nan=True)
    assert (out1["is_anomaly"] == out2["is_anomaly"]).all() and (out1["threshold_mode"] == "dynamic").all()
    assert out1["threshold"].iloc[30 + 500:].notna().all()  # per-position epsilon after the ignored start


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    import torch

    from src.models.lstm import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cuda") == "cpu" and resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


# ---------- pipeline: telemanom method, sharding, derived run ----------
@pytest.fixture
def tm_config(tiny_config):
    cfg = dict(tiny_config)
    cfg["methods"] = ["baseline", "telemanom"]
    cfg["telemanom"] = {"raw_units": True, "window": 30, "n_predictions": 3, "hidden_size": 8, "num_layers": 1,
                        "epochs": 2, "max_train_windows": 300, "smoothing_span": 20, "warmup_fill": 30,
                        "restore_best": False, "min_delta": 3e-4}
    cfg["dynamic_threshold"] = {"batch_size": 70, "window_size": 30, "p": 0.13, "l_s": 30}
    return cfg


def test_telemanom_pipeline_shards_and_evaluates(tm_config):
    from src.evaluation.evaluate import evaluate_run
    from src.pipeline import run_training

    reports = run_training(tm_config, shard=(0, 2))
    assert not (reports / "training_summary.json").exists()             # a shard never writes the summary
    assert len(list((reports / "scores").glob("*.json"))) == 2         # channels 0 and 2 of 3
    run_training(tm_config, resume=True)                                # collects + finishes the remaining channel
    summary = json.loads((reports / "training_summary.json").read_text())
    assert summary["failures"] == [] and summary["n_channels"] == 3
    info = summary["channels"][0]["methods"]["telemanom"]
    assert info["threshold_mode"] == "dynamic" and 0 <= info["val_flag_rate"] <= 1
    assert info["device"] == "cpu" and info["train_device"] == "cpu"      # where training actually ran
    t = pd.read_parquet(reports / "scores" / "S-1.parquet")
    assert {"telemanom_threshold", "telemanom_forecast"} <= set(t.columns)
    v = pd.read_parquet(reports / "scores" / "S-1.val.parquet")
    assert {"telemanom_forecast", "value"} <= set(v.columns)
    metrics = evaluate_run(reports)
    o = metrics["results"]["telemanom"]["overall"]
    assert o["threshold_mode"] == "dynamic" and metrics["results"]["telemanom"]["sensitivity"] == []
    assert o["false_alarms_per_10k"] is not None
    ev = pd.read_csv(reports / "events.csv")
    hit = ev[(ev.method == "telemanom") & ev.detected]
    assert (hit["decision_latency"] >= hit["latency"]).all()


def test_shard_argument_validation():
    from src.pipeline import parse_shard

    assert parse_shard("1/4") == (1, 4) and parse_shard(None) is None
    for bad in ("4/4", "-1/2", "1/0"):
        with pytest.raises(ValueError):
            parse_shard(bad)


def test_derived_lstm_dynamic_reuses_saved_models(tiny_config):
    from src.evaluation.evaluate import evaluate_run
    from src.pipeline import run_derived, run_training

    run_training(tiny_config, ["baseline", "lstm"])
    cfg = dict(tiny_config, run_name="pytest_dyn", derived_from="pytest", methods=["lstm_dynamic"],
               lstm_dynamic={"smoothing_span": 20, "warmup_fill": 30},
               dynamic_threshold={"batch_size": 70, "window_size": 30, "l_s": 30})
    out = run_derived(cfg)
    s = json.loads((out / "training_summary.json").read_text())
    assert s["failures"] == [] and s["derived_from"] == "pytest"
    assert all(c["methods"]["lstm_dynamic"]["forecast_max_abs_diff"] < 1e-5 for c in s["channels"])
    m = evaluate_run(out)
    assert m["results"]["lstm_dynamic"]["overall"]["event_tp"] + m["results"]["lstm_dynamic"]["overall"]["event_fn"] == 5


def test_channel_too_short_for_the_model_counts_as_missed_not_excluded(tm_config):
    """D-12 (312 training samples) cannot form one l_s = 250 training window: the method raises no alarm there and the
    channel stays in the evaluation, so all methods are compared on the same channels."""
    from src.evaluation.evaluate import evaluate_run
    from src.pipeline import run_training

    tm_config["telemanom"].update(window=1195, n_predictions=10)       # fit part is 1200 samples
    reports = run_training(tm_config)
    s = json.loads((reports / "training_summary.json").read_text())
    info = s["channels"][0]["methods"]["telemanom"]
    assert s["failures"] == [] and info["insufficient_data"] and info["calibratable"] is False
    t = pd.read_parquet(reports / "scores" / "S-1.parquet")
    assert t["telemanom_pred"].sum() == 0 and t["telemanom_score"].isna().all()
    m = evaluate_run(reports)
    assert m["n_channels_evaluated"] == 3
    o = m["results"]["telemanom"]["overall"]
    assert o["event_tp"] == 0 and o["event_fn"] == 5 and o["event_fp"] == 0
