"""Historical replay: only information available at time t, and the end state equals the offline result."""
import numpy as np
import pandas as pd

from src.models.dynamic_threshold import dynamic_threshold, release_times
from src.replay import REPLAY_LABEL, replay_state


def _table(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    e = np.abs(0.03 * rng.standard_normal(n))
    e[3000:3060] += 1.5
    e[4100:4120] += 1.2
    e_s = pd.Series(e).ewm(span=105).mean().to_numpy().copy()
    e_s[:250] = np.nan
    label = np.zeros(n, np.int8)
    label[3000:3100] = 1
    values = 0.6 * np.sin(2 * np.pi * t / 300)
    pred, _ = dynamic_threshold(e_s, values)
    static = (np.nan_to_num(e_s, nan=0) > 0.5).astype(np.int8)
    return pd.DataFrame({"timestep": t, "value": values, "label": label, "telemanom_score": e_s,
                         "telemanom_pred": pred, "lstm_score": e_s, "lstm_pred": static})


def test_label_text():
    assert REPLAY_LABEL == "Historical replay — not live telemetry"


def test_dynamic_replay_end_state_equals_offline_predictions():
    t = _table()
    s = replay_state(t, "telemanom", len(t))
    np.testing.assert_array_equal(s.pred, t["telemanom_pred"].to_numpy())
    assert s.decided == len(t)
    assert "overlaps a labelled anomaly" in set(s.alerts.hindsight)


def test_dynamic_replay_never_shows_undecided_or_future_flags():
    t = _table()
    rel = release_times(t["telemanom_score"].to_numpy())
    for tt in (2000, 2400, 3020, 3100, 3300, 4115):
        s = replay_state(t, "telemanom", tt)
        assert len(s.pred) == tt
        flagged = np.flatnonzero(s.pred)
        assert (rel[flagged] < tt).all()                    # every visible flag had been released by tt
        assert (s.alerts.raised_at < tt).all() and (s.alerts.delay >= 0).all()
    early = replay_state(t, "telemanom", 3030)              # the burst starts at 3000 but its batch is still open...
    later = replay_state(t, "telemanom", 3200)              # ...and is visible once the batch has closed
    assert early.pred[3000:3030].sum() <= later.pred[3000:3030].sum()
    assert later.pred[3000:3060].any()


def test_static_replay_is_immediate():
    t = _table()
    s = replay_state(t, "lstm", 3010)
    np.testing.assert_array_equal(s.pred, t["lstm_pred"].to_numpy()[:3010])
    assert s.decided == 3010 and (s.alerts.delay == 0).all()
    assert len(replay_state(t, "lstm", 0).pred) == 0
