"""Nonparametric dynamic thresholding with pruning (Hundman et al., KDD 2018, Telemanom).

Label-free: it looks only at the smoothed prediction errors ``e_s`` and the telemetry values themselves.
Nothing here reads a label, so it can be applied to test errors without violating the evaluation protocol.

Procedure (parameter defaults are the published Telemanom config, fixed before any run here):

* The error stream is cut into windows of ``window_size * batch_size`` (30 x 70 = 2,100) errors that advance by
  ``batch_size`` (70). Window 0 decides all of its own positions (except the first ``num_to_ignore``); every later
  window decides only its newest batch. A point's decision therefore uses at most ``batch_size - 1`` future errors:
  decisions are released once per batch, which is what the historical replay shows.
* Per window, ``epsilon = mean(e_s) + z * std(e_s)`` with z chosen from 2.5..11.5 to maximise
  ``(Δmean/mean + Δstd/std) / (n_sequences² + n_anomalous_points)`` (Δ = drop after removing points above epsilon),
  subject to <= 5 sequences and < 50 % of the window flagged. Default z = 12 when nothing qualifies.
  The same is done on errors mirrored around their mean (``inverse``: unusually *small* errors).
* A window is skipped when its errors are tiny relative to the telemetry's spread (the 5 % rules below).
* Flagged points are buffered by ``error_buffer`` on each side and grouped into sequences.
* Pruning: sequence maxima are sorted with the largest non-anomalous error appended; walking down, any sequence
  below the last relative drop of at least ``p`` (0.13) is discarded as not clearly separated from normal errors.
* Final anomalies are the union over windows; single-point groups are dropped.

Differences from the reference code, all deliberate and documented in docs/methodology.md:
* the last, partial window decides every position after the previous window instead of only its last 70
  (the reference leaves a gap of up to 69 undecided points);
* channels shorter than one window are treated as a single window (the reference adjusts the window count but not
  its length).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class DynamicThresholdConfig:
    batch_size: int = 70
    window_size: int = 30
    error_buffer: int = 100
    p: float = 0.13
    sd_lim: float = 12.0
    l_s: int = 250              # only used for num_to_ignore in the first window (Telemanom convention)
    min_relative: float = 0.05  # the "5 %" skip / magnitude rules
    min_error: float = 0.05     # absolute floor on max(e_s) in benchmark units ([-1, 1] scaled telemetry)

    @classmethod
    def from_dict(cls, d: dict | None) -> "DynamicThresholdConfig":
        d = dict(d or {})
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def to_dict(self) -> dict:
        return asdict(self)


def _dilate(mask: np.ndarray, k: int) -> np.ndarray:
    """True wherever a True lies within k positions (O(n) via a running sum)."""
    if k <= 0 or not mask.any():
        return mask.copy()
    n = mask.size
    cs = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
    i = np.arange(n)
    return (cs[np.minimum(i + k + 1, n)] - cs[np.maximum(i - k, 0)]) > 0


def _groups(idx: np.ndarray) -> list[np.ndarray]:
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)


def _multi(groups: list[np.ndarray]) -> list[np.ndarray]:
    return [g for g in groups if g[0] != g[-1]]


def find_epsilon(e_s: np.ndarray, cfg: DynamicThresholdConfig, inverse: bool = False) -> tuple[float, float]:
    """(epsilon, z) for one window. ``inverse`` mirrors errors around their mean."""
    mean, sd = float(e_s.mean()), float(e_s.std())
    x = 2 * mean - e_s if inverse else e_s
    best_eps, best_z, max_score = mean + cfg.sd_lim * sd, cfg.sd_lim, -1e7
    if sd == 0.0 or mean == 0.0:
        return best_eps, best_z
    for z in np.arange(2.5, cfg.sd_lim, 0.5):
        eps = mean + z * sd
        above = x >= eps
        if not above.any():
            continue
        pruned = x[~above]
        idx = np.flatnonzero(_dilate(above, cfg.error_buffer - 1))   # reference uses offsets 1..buffer-1 here
        n_seq = len(_multi(_groups(idx)))
        if pruned.size == 0:
            continue
        score = ((mean - pruned.mean()) / mean + (sd - pruned.std()) / sd) / (n_seq ** 2 + idx.size)
        if score >= max_score and n_seq <= 5 and idx.size < x.size * 0.5:
            max_score, best_eps, best_z = score, eps, float(z)
    return float(best_eps), best_z


def _compare(x: np.ndarray, e_s: np.ndarray, eps: float, values: np.ndarray, decided_from: int,
             flagged_before: np.ndarray, cfg: DynamicThresholdConfig) -> tuple[np.ndarray, float]:
    """Indices (window-local) flagged in the positions this window decides, and the largest non-anomalous error."""
    sd_values = float(values.std())
    inter_range = float(np.percentile(values, 95) - np.percentile(values, 5))
    e_max = float(e_s.max())
    empty = np.empty(0, dtype=np.int64)
    if not (e_s.std() > cfg.min_relative * sd_values or e_max > cfg.min_relative * inter_range) or not e_max > cfg.min_error:
        return empty, 0.0
    anom = (x >= eps) & (x > cfg.min_relative * inter_range)
    if not anom.any():
        return empty, 0.0
    idx = np.flatnonzero(_dilate(anom, cfg.error_buffer))
    idx = idx[idx >= decided_from]
    other = np.ones(x.size, dtype=bool)
    other[idx] = False
    other &= ~flagged_before
    non_anom_max = float(x[other].max()) if other.any() else 0.0
    return idx, non_anom_max


def _prune(x: np.ndarray, idx: np.ndarray, non_anom_max: float, p: float) -> np.ndarray:
    seqs = _multi(_groups(idx))
    if not seqs:
        return idx
    maxes = np.array([x[g[0] : g[-1] + 1].max() for g in seqs])
    order = np.argsort(-maxes, kind="stable")
    ladder = np.append(maxes[order], non_anom_max)
    remove: list[int] = []
    for i in range(len(ladder) - 1):
        drop = (ladder[i] - ladder[i + 1]) / ladder[i] if ladder[i] != 0 else 0.0
        if drop < p:
            remove.append(int(order[i]))
        else:
            remove = []
    keep = [g for j, g in enumerate(seqs) if j not in set(remove)]
    if not keep:
        return np.empty(0, dtype=np.int64)
    mask = np.zeros(x.size, dtype=bool)
    for g in keep:
        mask[g[0] : g[-1] + 1] = True
    return idx[mask[idx]]


def window_bounds(n: int, cfg: DynamicThresholdConfig) -> list[tuple[int, int, int]]:
    """(start, end, decided_from) per window, window-local ``decided_from``. Depends on n only via the last window."""
    bs, h = cfg.batch_size, cfg.batch_size * cfg.window_size
    num_to_ignore = 2 * cfg.l_s if n >= 2500 else (cfg.l_s if n >= 1800 else 0)
    if n <= h:
        return [(0, n, min(num_to_ignore, max(n - 1, 0)))]
    n_windows = (n - h) // bs
    out = [(0, h, num_to_ignore)]
    for i in range(1, n_windows + 1):
        out.append((i * bs, h + i * bs, h - bs))
    last_end = out[-1][1]
    if last_end < n:                                    # partial final window: decide everything after last_end
        start = n - h
        out.append((start, n, last_end - start))
    return out


def dynamic_threshold(e_s: np.ndarray, values: np.ndarray, cfg: DynamicThresholdConfig | dict | None = None,
                      decided_until: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Binary predictions and the per-position epsilon for a smoothed-error stream.

    ``e_s`` and ``values`` are aligned arrays (NaN = warm-up, not scored; leading NaNs only). Errors must be in the
    benchmark's units for the absolute 0.05 rule to mean what it means in the paper.
    ``decided_until``: if given, only windows that end at or before this index are processed - used by the replay to
    show exactly what was known at that moment.
    """
    cfg = cfg if isinstance(cfg, DynamicThresholdConfig) else DynamicThresholdConfig.from_dict(cfg)
    e_s, values = np.asarray(e_s, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if e_s.shape != values.shape:
        raise ValueError(f"e_s {e_s.shape} and values {values.shape} must be aligned")
    pred = np.zeros(e_s.size, dtype=np.int8)
    eps_out = np.full(e_s.size, np.nan)
    ok = np.flatnonzero(np.isfinite(e_s))
    if ok.size == 0:
        return pred, eps_out
    off = int(ok[0])
    if not np.isfinite(e_s[off:]).all():
        raise ValueError("NaN scores are only allowed as a leading warm-up")
    e, v = e_s[off:], values[off:]
    n = e.size
    flagged = np.zeros(n, dtype=bool)
    eps_local = np.full(n, np.nan)
    limit = None if decided_until is None else decided_until - off
    for start, end, dec in window_bounds(n, cfg):
        if limit is not None and end > limit:
            break
        w, wv = e[start:end], v[start:end]
        kept = []
        for inverse in (False, True):
            eps, _ = find_epsilon(w, cfg, inverse)
            if not inverse:
                eps_local[start + dec : end] = eps
            x = 2 * w.mean() - w if inverse else w
            idx, nam = _compare(x, w, eps, wv, dec, flagged[start:end], cfg)
            if idx.size:
                kept.append(_prune(x, idx, nam, cfg.p))
        if kept:
            flagged[start + np.unique(np.concatenate(kept))] = True
    final = np.zeros(n, dtype=bool)
    for g in _multi(_groups(np.flatnonzero(flagged))):
        final[g[0] : g[-1] + 1] = True
    pred[off:] = final
    eps_out[off:] = eps_local
    return pred, eps_out


def last_decided_index(n_scored: int, cfg: DynamicThresholdConfig | dict | None = None) -> int:
    """How many scored positions have a final decision when ``n_scored`` errors have arrived (replay)."""
    cfg = cfg if isinstance(cfg, DynamicThresholdConfig) else DynamicThresholdConfig.from_dict(cfg)
    h, bs = cfg.batch_size * cfg.window_size, cfg.batch_size
    if n_scored < h:
        return 0
    return h + ((n_scored - h) // bs) * bs


def release_times(scores: np.ndarray, cfg: DynamicThresholdConfig | dict | None = None) -> np.ndarray:
    """Timestep at which the decision about each position becomes available (the end of the window deciding it).
    Positions that are never scored get their own index."""
    cfg = cfg if isinstance(cfg, DynamicThresholdConfig) else DynamicThresholdConfig.from_dict(cfg)
    s = np.asarray(scores, dtype=np.float64)
    rel = np.arange(s.size)
    ok = np.flatnonzero(np.isfinite(s))
    if ok.size == 0:
        return rel
    off = int(ok[0])
    for start, end, dec in window_bounds(s.size - off, cfg):
        rel[off + start + dec : off + end] = off + end - 1
    return rel
