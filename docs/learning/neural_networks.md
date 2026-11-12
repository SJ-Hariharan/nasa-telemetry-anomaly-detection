# Neural networks, through this project's LSTM training loop

Everything below is tied to three files: `src/models/lstm.py` (`LSTMForecaster`, `LSTMDetector.fit`),
`configs/default.yaml` (`lstm:` section) and `src/utils/reproducibility.py` (`set_seed`).
All numbers in the worked examples were checked with `.venv/bin/python` (and where possible with the
project's own functions). For results of the full experiment, see `EXPERIMENTS.md`; nothing here
reports them.

The task the network solves, in one line: given the last $W=100$ timesteps of one channel
(value + one-hot command flags, $D=25$ features for SMAP, 55 for MSL), predict the next scaled value $x_t$.

| Concept | Where in the code | Config key (default) |
|---|---|---|
| Linear layer | `self.head = nn.Linear(hidden_size, 1)` | `hidden_size: 64` |
| Loss | `loss_fn = nn.MSELoss()` | – |
| Optimiser | `torch.optim.Adam(..., lr=self.learning_rate)` | `learning_rate: 2.0e-3` |
| Mini-batches / epochs | `for i in range(0, len(order), self.batch_size)` | `batch_size: 128`, `epochs: 60` |
| Subsampling | `np.linspace(0, len(Xtr) - 1, self.max_train_windows)` | `max_train_windows: 4000` |
| Dropout | `nn.LSTM(dropout=...)`, `nn.Dropout(dropout)` | `dropout: 0.3` |
| Clipping | `nn.utils.clip_grad_norm_(..., 1.0)` | hard-coded 1.0 |
| Early stopping | `bad_epochs >= self.patience`, `best_state` | `patience: 6` |
| Seeds | `set_seed(cfg["seed"])` in `src/pipeline.py` | `seed: 42` |

---

## 1. Neurons, linear layers and activations

**Problem.** We need a function with adjustable knobs that maps numbers in to a number out.
**Why.** A fixed formula (e.g. "next value = last value") cannot adapt to 81 different channels; learnable weights can.

**Concept.** A neuron computes a weighted sum then (optionally) a non-linearity:
$z = \mathbf{w}^\top \mathbf{x} + b$, $a = \phi(z)$, where $\mathbf{x}\in\mathbb{R}^D$ is the input,
$\mathbf{w}\in\mathbb{R}^D$ the weights, $b$ the bias, $\phi$ the activation. A *linear layer* is many neurons
side by side: $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ with $W\in\mathbb{R}^{\text{out}\times\text{in}}$.
Without $\phi$, stacked linear layers collapse into one linear map, so non-linearities are what give depth its power.
Common $\phi$: sigmoid $\sigma(z)=1/(1+e^{-z})\in(0,1)$ (used as "gates" inside the LSTM), $\tanh(z)\in(-1,1)$
(used for LSTM candidate/cell output), ReLU $\max(0,z)$ (not used here).

**Example.** $\mathbf{x}=(0.5,-1.0)$, $\mathbf{w}=(0.4,0.3)$, $b=0.2$: $z=0.2-0.3+0.2=0.1$, $\tanh(0.1)=0.0997$.

**In the code.** The output layer has *no* activation, because the target is an unbounded real number (a z-scored value):
```python
self.head = nn.Linear(hidden_size, 1)                 # 64 -> 1, i.e. 64 weights + 1 bias
return self.head(self.dropout(last)).squeeze(-1)      # (batch,)
```
The sigmoid/tanh activations live inside `nn.LSTM` (see `lstm_from_first_principles.md`).

**Alternatives.** A plain MLP over lagged values (the Isolation Forest in `src/models/detectors.py` uses lag features instead of a network).
**Limitations.** A bounded output activation (tanh) would clip large forecasts; leaving the head linear avoids that.

**Interview checkpoint**
- *Why no activation on the final layer?* Regression to an unbounded target; an activation would restrict the output range.
- *What happens if you stack linear layers with no activation?* $W_2(W_1x+b_1)+b_2$ is still linear — no extra expressiveness.
- *Why sigmoid for gates?* Its $(0,1)$ range reads as "fraction to let through".

---

## 2. Loss (MSE), gradient descent and backpropagation

**Problem.** We need one number that says how wrong the forecasts are, and a rule to change weights to reduce it.
**Why.** Training = minimising that number over the training windows.

**Concept.** Mean squared error over a batch of $B$ windows: $L=\frac{1}{B}\sum_{j=1}^{B}(\hat{x}_j-x_j)^2$,
$\hat{x}_j$ = forecast, $x_j$ = true next value. Squaring punishes large errors more and is smooth (differentiable).
Gradient descent updates every parameter $\theta$ by $\theta \leftarrow \theta - \eta\,\partial L/\partial\theta$ ($\eta$ = learning rate).
*Backpropagation* computes all $\partial L/\partial\theta$ efficiently with the chain rule, from the loss backwards.

**Example (MSE).** forecasts $(0.2,0.5,0.9)$, targets $(0.0,0.5,1.0)$: $(0.04+0+0.01)/3=0.01667$ (same as `nn.MSELoss`).
**Example (one GD step).** $\hat{x}=wx$, $x=2$, target $4$, $w=1$: $L=(2-4)^2=4$, $\partial L/\partial w=2(wx-y)x=-8$.
With $\eta=0.1$: $w=1.8$, new $L=0.16$.
**Example (backprop through tanh).** $\hat{y}=\tanh(wx+b)$, $w=0.5,x=1,b=0,y=1$:
$z=0.5$, $a=0.4621$, $\partial L/\partial a=2(a-y)=-1.0758$, $\partial a/\partial z=1-a^2=0.7864$,
$\partial L/\partial w=-1.0758\times0.7864\times1=-0.8460$ (PyTorch autograd gives the same).

**In the code.**
```python
opt.zero_grad()                       # gradients accumulate in PyTorch, so reset each step
loss = loss_fn(self.model(xb), yb)    # forward pass + MSE
loss.backward()                       # backpropagation (autograd)
opt.step()                            # parameter update (Adam, below)
```
**Alternatives.** MAE/Huber loss are less sensitive to outliers in training data (useful if train contains unlabelled anomalies).
**Limitations.** MSE is pulled towards outliers; if the train file contains unlabelled glitches, the model partly learns them.

**Interview checkpoint**
- *Why `zero_grad()`?* `.backward()` adds to `.grad`; without resetting you'd sum gradients across batches.
- *Why is the anomaly score $|e_t|$ but the loss $e_t^2$?* Squared loss is convenient for optimisation; for scoring, absolute error keeps the scale interpretable (same units as the value).
- *What does backprop actually return?* $\partial L/\partial\theta$ for every parameter, by the chain rule.

---

## 3. Adam

**Problem.** One global learning rate fits some parameters badly (gradients differ by orders of magnitude).
**Why.** Adam adapts the step per parameter, so it trains LSTMs reliably with little tuning.

**Concept.** With gradient $g_t$: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ (momentum), $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$
(scale), bias-corrected $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$, update
$\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. Defaults $\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$.

**Example.** Same toy problem, $g_1=-8$, $\eta=2\times10^{-3}$ (our config): $m_1=-0.8$, $v_1=0.064$,
$\hat m_1=-8$, $\hat v_1=64$, step $=0.002\times(-8/8)=-0.002$ ⇒ $w=1.002$ (verified with `torch.optim.Adam`).
Note: the first step size is ≈ $\eta$ regardless of the gradient's magnitude.

**In the code.** `opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)` with `learning_rate: 2.0e-3`.
**Alternatives.** SGD+momentum (often generalises well but needs LR tuning), AdamW (decoupled weight decay), LR schedules.
**Limitations.** No weight decay or LR schedule here; with early stopping that is acceptable for a small model.

**Interview checkpoint**
- *Why bias correction?* $m,v$ start at 0, so early averages are biased towards 0.
- *What does $\sqrt{\hat v}$ do?* Normalises step size per parameter.
- *Would you tune LR per channel?* Not without a validation protocol per channel; the project uses one config for all.

---

## 4. Mini-batches, epochs, shuffling and subsampling

**Problem.** Computing the gradient on all windows each step is slow; on one window it is noisy.
**Why.** Mini-batches give a cheap, reasonably accurate gradient estimate; an *epoch* = one pass over all training windows.

**Concept.** Batches of size $B$ ⇒ $\lceil N/B\rceil$ updates per epoch. Shuffling the order each epoch decorrelates consecutive batches.
Crucially, the project shuffles **windows**, not **timesteps**: each sample is an intact 100-step sequence, so temporal order *inside* a window is preserved — only the order in which windows are visited changes.

**Example.** A train file of 2,000 steps, `val_fraction: 0.2` ⇒ `split = 1600`. `forecast_windows(arr, 100)` gives
1,900 windows (targets $t=100..1999$): 1,500 with target $<1600$ (train), 400 validation. If $N=4000$, $B=128$:
32 batches per epoch (31 full + one of 32). Strided subsampling `np.linspace(0, 9, 4).round()` keeps `[0, 3, 6, 9]` — evenly spread over the whole period.

**In the code** (`LSTMDetector.fit`):
```python
tr, va = t_idx < split_idx, t_idx >= split_idx           # split by TARGET timestep
if len(Xtr) > self.max_train_windows:  # even stride keeps coverage of the whole period
    keep = np.linspace(0, len(Xtr) - 1, self.max_train_windows).round().astype(int)
order = torch.randperm(len(Xtr), generator=gen).numpy()  # shuffle WINDOWS, not timesteps
```
**Alternatives.** Random subsampling (could cluster in one regime), using all windows (more CPU), stateful RNN training over the full series.
**Limitations.** Strided subsampling drops near-duplicate overlapping windows, but a rare regime may still be thinned out. Validation windows are *not* subsampled.

**Interview checkpoint**
- *Why is shuffling windows not leakage?* Each window only contains its own past; train windows all have targets before the split.
- *Why split by target index?* A train window's inputs then also lie before the split; validation windows may look back into the fit region (past data, legitimate) but are never trained on.
- *Why linspace instead of random choice?* Deterministic and covers the whole timeline evenly.

---

## 5. Overfitting, train vs validation loss, early stopping

**Problem.** A network can memorise training windows and forecast new data badly.
**Why.** For anomaly detection, an over-fitted model has tiny errors on train and inflated errors on normal test data ⇒ false alarms.

**Concept.** Track validation loss (data not trained on) each epoch. Keep the weights from the epoch with the lowest validation loss; stop after `patience` epochs without improvement (by more than $10^{-6}$).
Train loss here is computed in `train()` mode (dropout on) and averaged *during* the epoch, while validation loss is in `eval()` mode — so train loss above validation loss is normal, not a bug.

**Example** (simulated with the exact `fit` logic, `patience=6`): val losses
`0.50, 0.30, 0.22, 0.21, 0.215, 0.209, 0.2095, 0.212, 0.211, 0.213, 0.214, 0.216` ⇒ best epoch 5 (0.209), six non-improving epochs (6–11) ⇒ stop after epoch 11, restore epoch-5 weights.

**In the code.**
```python
if val_loss < best_val - 1e-6:
    best_val, best_state, bad_epochs, self.best_epoch = val_loss, copy.deepcopy(self.model.state_dict()), 0, epoch
else:
    bad_epochs += 1
    if bad_epochs >= self.patience: break
self.model.load_state_dict(best_state)  # early stopping: keep the best validation epoch
```
`copy.deepcopy` matters: `state_dict()` returns references to live tensors that later steps would overwrite.
`best_epoch`, `history` are saved to `meta.json` by `src/models/persistence.py`.

**Alternatives.** Fixed epoch count; weight decay; smaller model.
**Limitations.** The same validation split is used for early stopping **and** for threshold calibration (`src/pipeline.py`), so validation errors are slightly optimistic and thresholds slightly low.

**Interview checkpoint**
- *Why restore the best state instead of the last?* The last `patience` epochs were, by definition, worse on validation.
- *Can validation loss be lower than train loss?* Yes — dropout is active only in training, and train loss is averaged while weights are still changing.
- *What is the risk of reusing validation for early stopping and thresholds?* Mild optimistic bias in the calibrated threshold.

---

## 6. Dropout and gradient clipping

**Problem.** (a) Co-adapted neurons overfit; (b) RNN gradients occasionally explode and one bad step ruins training.
**Why.** Dropout is cheap regularisation; clipping is cheap insurance.

**Concept.** *Dropout* with rate $p$ zeros each activation with probability $p$ during training and scales survivors by $1/(1-p)$ so the expected value is unchanged; at inference (`eval()`) it is off.
*Gradient-norm clipping*: if $\|\mathbf g\|_2 > c$, rescale $\mathbf g \leftarrow \mathbf g\cdot c/\|\mathbf g\|_2$ (direction kept, length capped).

**Example.** $p=0.3$: survivors become $1/0.7=1.4286$ (seen with `nn.Dropout(0.3)` on a vector of ones).
Clipping $\mathbf g=(3,4)$, $\|\mathbf g\|=5$, $c=1$ ⇒ $(0.6, 0.8)$; `clip_grad_norm_` returns the *pre-clip* norm 5.0.

**In the code.** Dropout appears twice: `nn.LSTM(..., dropout=dropout if num_layers > 1 else 0.0)` applies it *between* LSTM layers (PyTorch warns if used with one layer, hence the guard), and `self.dropout = nn.Dropout(dropout)` before the head. Clipping: `nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)` — the norm is computed over **all** parameters jointly.

**Alternatives.** Recurrent (variational) dropout, weight decay, clipping by value (`clip_grad_value_`).
**Limitations.** `configs/smoke.yaml` sets `num_layers: 1`, so there inter-layer dropout is disabled and only head dropout remains.

**Interview checkpoint**
- *Why scale by $1/(1-p)$?* Keeps the expected activation equal between train and inference.
- *Does clipping fix vanishing gradients?* No, only exploding ones; LSTM gating addresses vanishing.
- *Why must you call `model.eval()` before scoring?* To disable dropout; `_predict_windows` does this.

---

## 7. Seeds and determinism

**Problem.** Weight init, dropout masks and shuffling are random; two runs could give different thresholds and metrics.
**Why.** A result you cannot reproduce cannot be debugged or compared fairly.

**Concept.** Seed every random number generator; ask PyTorch to use deterministic kernels.
**Example.** `torch.manual_seed(0); torch.randn(2)` gives the same two numbers on every call sequence on the same machine.
**In the code.**
```python
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
torch.use_deterministic_algorithms(True, warn_only=True)
```
`src/pipeline.py` calls `set_seed(cfg["seed"])` **per channel and per method**, so channel P-1's result does not depend on which channels ran before it. `LSTMDetector.fit` re-seeds with its own `seed` (passed from `cfg["seed"]`): `torch.manual_seed(self.seed)` for weight init and dropout, and a separate `torch.Generator().manual_seed(self.seed)` for the window shuffling order, so changing the config seed changes both.
**Alternatives.** Run several seeds and report mean ± std (stronger evidence, more CPU).
**Limitations.** As the docstring says, bit-exact equality across hardware/OS/library versions is not guaranteed; `warn_only=True` means a non-deterministic op only warns.

**Interview checkpoint**
- *Why seed per channel?* Makes each channel's model independent of processing order and of which channels are selected.
- *Does a seed guarantee identical results on another laptop?* No — only close results; library versions and BLAS kernels differ.
- *Changing `seed:` in the config — what changes in the LSTM?* Initial weights and dropout masks, but not the window order (generator fixed at 0).

---

## Exercises

1. Run `python -m src.cli all --config configs/smoke.yaml`, then open `models/smoke/lstm/P-1/meta.json` and plot `history.train_loss` vs `history.val_loss`. Which epoch is `best_epoch`, and why can train loss be above val loss?
2. Copy `configs/smoke.yaml` to `configs/my_smoke.yaml` (keep `inherit: default.yaml`), set `run_name: my_smoke`, `lstm.epochs: 10`, `lstm.patience: 2`. Re-run and check `epochs_run` in `reports/runs/my_smoke/training_summary.json`.
3. In a Python shell, build `LSTMForecaster(25, 16, 1, 0.3)` and confirm it has 2,769 parameters; explain each tensor's shape.
4. Write a 10-line script that trains `LSTMDetector(window=20, hidden_size=8, num_layers=1, epochs=3)` on a sine wave twice with `set_seed(1)` and compares the predictions (cf. `test_lstm_forward_shape_and_learning` in `tests/test_models_and_metrics.py`).
5. Temporarily (in a scratch copy, not the repo) remove `clip_grad_norm_` and set `learning_rate: 0.1`. What happens to the loss curve?
