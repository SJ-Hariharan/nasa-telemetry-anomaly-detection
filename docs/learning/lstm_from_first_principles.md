# LSTMs from first principles — and how this project uses one to detect anomalies

Code referenced: `src/models/lstm.py` (`LSTMForecaster`, `LSTMDetector.predict/score`),
`src/features/windows.py` (`forecast_windows`), `src/models/thresholds.py`, `configs/default.yaml`.
Numbers in examples were verified with `.venv/bin/python` (the LSTM step was checked against `torch.nn.LSTM`).
Results of the full run: see `EXPERIMENTS.md`.

---

## 1. The problem: "what should the next value be?"

**Problem.** For each channel we want a model of *normal* behaviour, so we can flag where reality departs from it.
**Why a sequence model.** Telemetry depends on the recent past and on commands issued earlier (24 command flags for SMAP, 54 for MSL, one-hot in columns 1..). A model must carry information across many steps.

**Concept.** One-step-ahead forecasting: input $X_t=(\mathbf{x}_{t-W},\dots,\mathbf{x}_{t-1})$, $W=100$, each $\mathbf{x}\in\mathbb{R}^{D}$ ($D=25$ SMAP / $55$ MSL);
output $\hat{x}_t$ = predicted scaled value at $t$.

**Example.** `forecast_windows` docstring: values `[a,b,c,d]`, `window=2` ⇒ `X[0]=[a,b] → y[0]=c (t=2)`, `X[1]=[b,c] → y[1]=d (t=3)`.
With 2,000 steps and $W=100$ you get 1,900 windows, shape `(1900, 100, 25)`.

**In the code.**
```python
X = sliding_window_view(array, window, axis=0)[:-1].transpose(0, 2, 1)   # (n-W, W, f), no copy
target_idx = np.arange(window, n)                                          # which t each window predicts
```

---

## 2. Why plain RNNs struggle: vanishing (and exploding) gradients

**Concept.** A vanilla RNN: $\mathbf h_t=\tanh(W_x\mathbf x_t+W_h\mathbf h_{t-1}+\mathbf b)$. The gradient from step $t$ back to step $t-k$ multiplies $k$ Jacobians
$\prod \partial \mathbf h_{s}/\partial \mathbf h_{s-1}=\prod \mathrm{diag}(1-\mathbf h_s^2)\,W_h$. If their typical size is $<1$ the product shrinks exponentially; if $>1$ it explodes.

**Example.** Factor $0.9$ over our window of 100 steps: $0.9^{100}=2.7\times10^{-5}$; factor $0.5$: $7.9\times10^{-31}$; factor $1.1$: $1.4\times10^{4}$.
So a command 80 steps back barely influences learning in a vanilla RNN.

**Fixes.** Exploding → gradient clipping (`clip_grad_norm_(…, 1.0)` in `fit`). Vanishing → architectures with an additive memory path: the LSTM.

---

## 3. The LSTM cell

**Intuition.** Keep a separate *cell state* $\mathbf c_t$ (long-term memory) that is updated mostly by **addition**, controlled by three "valves" (gates) with values in $(0,1)$.

**Equations** (PyTorch's convention; $\odot$ = element-wise product):

$$\mathbf i_t=\sigma(W_{ii}\mathbf x_t+\mathbf b_{ii}+W_{hi}\mathbf h_{t-1}+\mathbf b_{hi})\quad\text{(input gate)}$$
$$\mathbf f_t=\sigma(W_{if}\mathbf x_t+\mathbf b_{if}+W_{hf}\mathbf h_{t-1}+\mathbf b_{hf})\quad\text{(forget gate)}$$
$$\mathbf g_t=\tanh(W_{ig}\mathbf x_t+\mathbf b_{ig}+W_{hg}\mathbf h_{t-1}+\mathbf b_{hg})\quad\text{(candidate)}$$
$$\mathbf o_t=\sigma(W_{io}\mathbf x_t+\mathbf b_{io}+W_{ho}\mathbf h_{t-1}+\mathbf b_{ho})\quad\text{(output gate)}$$
$$\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\mathbf g_t,\qquad \mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)$$

| Symbol | Meaning | Shape (first layer, $D=25$, $H=64$) |
|---|---|---|
| $\mathbf x_t$ | input at step $t$ (value + 24 command flags) | $(25,)$ |
| $\mathbf h_{t-1}$ | previous hidden state (short-term output) | $(64,)$ |
| $\mathbf c_{t-1}$ | previous cell state (long-term memory) | $(64,)$ |
| $W_{i\cdot}$ | input-to-hidden weights, one per gate | $(64,25)$ each; stacked `weight_ih_l0` $(256,25)$ |
| $W_{h\cdot}$ | hidden-to-hidden weights | $(64,64)$ each; stacked `weight_hh_l0` $(256,64)$ |
| $\mathbf b_{i\cdot},\mathbf b_{h\cdot}$ | two biases per gate (PyTorch keeps both) | $(64,)$ each; `bias_ih_l0`, `bias_hh_l0` $(256,)$ |
| $\mathbf i_t,\mathbf f_t,\mathbf o_t$ | gates in $(0,1)$ | $(64,)$ |
| $\mathbf g_t$ | candidate new content in $(-1,1)$ | $(64,)$ |

PyTorch stacks gates in the order **i, f, g, o** inside the 256 rows. In layer 2 the input is layer 1's $\mathbf h_t$, so `weight_ih_l1` is $(256,64)$.

**Why it fixes vanishing gradients.** $\partial\mathbf c_t/\partial\mathbf c_{t-1}=\mathrm{diag}(\mathbf f_t)$ (plus indirect terms). If the network learns $\mathbf f\approx1$, gradients pass through many steps nearly unchanged — no repeated squashing by $\tanh$ and $W_h$.

### Worked single step (hidden size 1, input size 1)

Weights $(w_x,w_h,b)$: input gate $(0.5,0.1,0)$, forget $(0.5,0.2,1.0)$, candidate $(1.0,0.3,0)$, output $(1.0,0.4,0)$.
State $h_{t-1}=0$, $c_{t-1}=0.5$, input $x_t=1.0$.

| Quantity | Computation | Value |
|---|---|---|
| $i_t$ | $\sigma(0.5)$ | 0.6225 |
| $f_t$ | $\sigma(0.5+1.0)=\sigma(1.5)$ | 0.8176 |
| $g_t$ | $\tanh(1.0)$ | 0.7616 |
| $o_t$ | $\sigma(1.0)$ | 0.7311 |
| $c_t$ | $0.8176\cdot0.5+0.6225\cdot0.7616$ | 0.8828 |
| $h_t$ | $0.7311\cdot\tanh(0.8828)$ | 0.5175 |

Loading these weights into `nn.LSTM(1, 1)` gives $h_t=0.517475$, $c_t=0.882849$ — identical.
A linear head $\hat x=0.8h_t+0.1$ would forecast $0.5140$.
Read it: the forget gate kept 82% of the old memory; the input gate wrote 62% of the candidate on top.

---

## 4. How `nn.LSTM` maps to this — and the readout

```python
self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
...
out, _ = self.lstm(x)               # out: (batch, window, hidden) - hidden state at every step
last = out[:, -1, :]                # summary of the whole window = hidden state at the last step
return self.head(self.dropout(last)).squeeze(-1)  # (batch,)
```
- `batch_first=True`: tensors are `(batch, time, features)`, e.g. `(128, 100, 25)`, matching `forecast_windows`.
- `num_layers=2`: layer 2 runs the same equations on layer 1's hidden sequence. `dropout=0.3` is applied to layer 1's outputs before layer 2 (not after the last layer).
- Initial $(\mathbf h_0,\mathbf c_0)$ default to zeros for every window: each window is processed independently (stateless).
- `out` has the top layer's $\mathbf h_t$ for all 100 steps; only the last, $\mathbf h_{t-1}$ after reading $\mathbf x_{t-1}$, is needed to forecast $x_t$. It has "seen" the whole window through the recurrence.
- `nn.Dropout` → `nn.Linear(64, 1)` → `squeeze` gives one number per window.

### Parameter count

Per layer: $4H(D_{\text{in}}+H)+8H$ (four gates; two bias vectors each). Instantiating
`LSTMForecaster(25, 64, 2, 0.3)` and summing `p.numel()`:

| Tensor | Shape | Count |
|---|---|---|
| `lstm.weight_ih_l0` | (256, 25) | 6,400 |
| `lstm.weight_hh_l0` | (256, 64) | 16,384 |
| `lstm.bias_ih_l0`, `bias_hh_l0` | (256,) ×2 | 512 |
| `lstm.weight_ih_l1`, `weight_hh_l1` | (256, 64) ×2 | 32,768 |
| `lstm.bias_ih_l1`, `bias_hh_l1` | (256,) ×2 | 512 |
| `head.weight`, `head.bias` | (1, 64), (1,) | 65 |
| **Total** | | **56,641** |

For an MSL channel ($D=55$): 64,321. Smoke config (`hidden_size: 16, num_layers: 1`, $D=25$): 2,769.
One model is trained **per channel**, on at most `max_train_windows: 4000` windows.

**Interview checkpoint**
- *Why does PyTorch have two bias vectors per gate?* cuDNN compatibility; mathematically they just add, so one is redundant.
- *What is in `out[:, -1, :]` vs `h_n`?* For the top layer they're the same vector; `h_n` also holds the last state of lower layers.
- *Why stateless windows?* Simple, shuffle-able, and inference on any 100-step history gives the same answer as in training.
- *How many parameters for D=55?* 64,321 — only `weight_ih_l0` grows ($256\times55$).

---

## 5. Forecasting vs reconstruction

| | One-step forecasting (this project) | Reconstruction autoencoder |
|---|---|---|
| Learns | $p(x_t\mid \text{past } W)$ | compress & rebuild a whole window |
| Error belongs to | exactly one timestep $t$ | a window (must be spread back to timesteps) |
| Causal? | yes — score at $t$ uses only $x_{\le t}$ | only if windows end at $t$ |
| Weakness | unpredictable-but-normal signals (noise) give high error | can reconstruct anomalies too well if it generalises |

**Why forecasting here.** Labels are timestep intervals; a forecast error is naturally aligned to $t$ (`pred[t_idx] = ...` in `LSTMDetector.predict`), which keeps label alignment simple and matches the original Telemanom method. It also obeys the project-wide rule in `src/features/windows.py`: *the score for $t$ may use $x_0..x_t$ only*.

---

## 6. From errors to anomaly scores to decisions

**Concept.**
1. Prediction error $e_t=|x_t-\hat x_t|$ on the scaled value (column 0).
2. Smoothing with an exponentially weighted moving average (EWMA), $\alpha=2/(\text{span}+1)$; span 30 ⇒ $\alpha=0.0645$.
   pandas' default (`adjust=True`) computes $s_t=\sum_{k\ge0}(1-\alpha)^k e_{t-k}\,/\sum_{k\ge0}(1-\alpha)^k$ over available past errors — causal.
   Smoothing suppresses single noisy misses and rewards *sustained* surprise.
3. Threshold per channel: `calibrate_threshold` = 0.995-quantile of validation scores × `margin` (1.0). Predict 1 if $s_t>$ threshold (`apply_threshold`; NaN warm-up → 0).

**Example.** $x=(0.20,0.25,0.90)$, $\hat x=(0.22,0.24,0.30)$ ⇒ $e=(0.02,0.01,0.60)$.
EWMA with span 3 ($\alpha=0.5$) on errors `[nan, nan, 0.1, 0.1, 0.1, 1.0, 0.1]` (project code path) gives
`[nan, nan, 0.1, 0.1, 0.1, 0.58, 0.3323]`: at the spike, $(0.1\cdot0.125+0.1\cdot0.25+0.1\cdot0.5+1.0)/1.875=0.58$, and it decays afterwards instead of vanishing immediately.

**In the code.**
```python
err = np.abs(array_scaled[:, VALUE_COL] - self.predict(array_scaled))
return pd.Series(err).ewm(span=self.smoothing_span, ignore_na=True).mean().where(~np.isnan(err)).to_numpy()
```
The first `window` steps have no forecast ⇒ NaN score ("warm-up").

**Limitations.** EWMA adds detection latency (a short spike needs to be large to lift the average) and makes alarms linger after the event — this creates extra false-alarm points right after true events. Validation scores are computed with `det.score(train_s)[split:]`, i.e. the EWMA there is warmed up on in-sample errors, whereas the test series starts fresh.

---

## 7. Comparison with the original Telemanom (Hundman et al., KDD 2018)

| | Telemanom | This project (`configs/default.yaml`) |
|---|---|---|
| Input length | `l_s = 250` | `window: 100` |
| Network | 2 LSTM layers `[80, 80]`, dropout 0.3 | 2 layers, `hidden_size: 64`, dropout 0.3 |
| Output | `n_predictions = 10` future values | 1 value (next step) |
| Threshold | nonparametric dynamic thresholding on smoothed errors, with pruning | fixed per-channel 0.995-quantile of validation scores |
| Training data | full train file | ≤ 4,000 evenly strided windows per channel |

**Trade-offs, honestly.** Shorter windows, fewer units and window subsampling cut CPU time (81 channels on a laptop CPU) but can miss long-range context. One-step forecasting is easier to align and explain, but gives the model fewer constraints than predicting 10 steps. A static validation-quantile threshold is simple, label-free and reproducible, but it does not adapt to local error levels the way dynamic thresholding does — expect more false alarms on noisy channels. This is therefore a *Telemanom-style* detector, not a reproduction; do not compare its numbers to the paper as like-for-like.

---

## 8. When an LSTM is NOT the right tool

- **Constant or near-constant channels** (19 channels are constant in train): any change is novel; a rule or the rolling z-score baseline is enough.
- **Very little training data**: 56k parameters on a few thousand points over-fits; use statistical baselines or Isolation Forest.
- **Pure point spikes**: a rolling z-score (`RollingZScoreDetector`) detects these cheaply and transparently.
- **Very long dependencies or many channels jointly**: consider temporal CNNs/Transformers or a multivariate model.
- **Strict latency/explainability requirements**: simpler models are easier to certify.

**Interview checkpoint**
- *Why EWMA and not a centred moving average?* A centred average uses future errors — not causal, leaks future into the score.
- *Why per-channel thresholds?* Error scales differ hugely between channels; one global cut would over-alert on noisy channels.
- *What is the main weakness of a static quantile threshold?* At q = 0.995, ~0.5% of nominal points are flagged by design; it cannot adapt to changing noise.
- *Why is this not a Telemanom reproduction?* Different window, size, horizon and thresholding.

---

## Exercises

1. In Python, instantiate `LSTMForecaster(25, 64, 2, 0.3)` and print `[(n, p.shape) for n, p in model.named_parameters()]`; reconcile each shape with the table above.
2. Reproduce the worked step: set the weights of `nn.LSTM(1, 1)` (gate order i, f, g, o) and check $h_t=0.5175$, $c_t=0.8828$. Then set the forget bias to $-5$; what happens to $c_t$?
3. After `python -m src.cli all --config configs/smoke.yaml`, load `reports/runs/smoke/scores/P-1.parquet` and plot `value`, `lstm_forecast` and `lstm_score` for timesteps 0–2000.
4. Compute the score for the same run with `smoothing_span` 1 and 30 (by calling `LSTMDetector.score` on a loaded bundle after changing `det.smoothing_span`). How do latency and false alarms change?
5. List the channels with constant train data (see `reports/data_validation.json`) and argue which detector you would deploy for them.
