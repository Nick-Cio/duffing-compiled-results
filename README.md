# Learning the Forced Duffing Oscillator with Neural ODEs

Nick Ciordas — SURF 2026 (warm-up study for the aeroelastic dataset)

All experiments learn the vector field of the forced Duffing oscillator

    x'' + δx' − x + x³ = γ cos(ωt),      state s = (x, ẋ)

from simulated trajectories, via one-step training: predict s(t + 0.01 s) from s(t)
through a single RK4 step, minibatch Adam on MSE. The study is a ladder — each step
adds one ingredient and asks one question.

## Package layout

```
duffing_compiled_results/
├── README.md            ← this file
├── requirements.txt     ← jax, numpy, matplotlib, jupyter (Python 3.13 venv)
├── notebooks/           ← 5 executed notebooks, ALL results & figures included
├── scripts/             ← the same code as standalone .py scripts (see note below)
└── data/                ← trained-weight caches (params_*.npz), so nothing retrains
```

**Notebooks are the primary artifact** — every figure and printed table in them is a
saved output, viewable without running anything. The scripts in `scripts/` are
extracted verbatim from the notebook code cells (cell-for-cell, markdown converted to
comments), so "the script that generated result X" is exactly the correspondingly
named script. Merged notebooks contain several experiments as Parts; each Part became
its own script.

## The ladder — what each step asks and finds

| # | script / notebook | question | headline result |
|---|---|---|---|
| 1 | `step01` (nb 1, Part A) | Can a 4.6k-param MLP learn one oscillator's vector field from a single trajectory? | Yes on-attractor (rollout err ~3e-3 over 20 s), but an unseen IC lands in the **wrong well** (err ~1.7) |
| 2 | `step02` (nb 1, Part B) | Does a 9×7 IC grid fix that? | Yes — all held-out ICs pass; trust region extends well past the IC box; breakdown = wrong-well flips |
| 3 | `step03` (nb 2) | Is enforcing the system's odd symmetry worth it? Where does the data actually live? | Symmetry buys nothing with two-sided data, everything with one-sided (wrong wells 21/28 → 0/28). Breakdown tracks **data coverage**, not the IC box |
| 4 | `step04` (nb 3) | Does the additive split a = f(x) + d(ẋ) + g(phase) beat one black box? | Yes — wrong-well ray starts 23→5 and 17→1; d and g recovered essentially exactly; f matches x − x³ in-range but **tanh saturates vs the cubic** outside |
| 5 | `step05` (nb 4, Part A) | Does additivity survive a violent cross-well regime (γ=4)? | Yes — ~10× lower ray error; both learned limit cycles sit exactly on the true orbit |
| 6a | `step06a` (nb 4, Part B) | Can the models hold a freshly-doubled **period-8** orbit? | Both hold exact period-8 (strobe residual ~1e-11) incl. the 0.02-scale pair splitting |
| 6b | `step06b` (nb 4, Part C) | Period-3 coexisting with chaos — do the models keep both attractors? | Both hold period-3 exactly; but **additive collapses the chaotic attractor to period-4** while scalar preserves the dust. Summary stats missed it; only plotting the invariant set caught it |
| 7 | `step07` (nb 5, Part A) | Feed δ, γ, ω to the network — can one model learn the whole family? | Yes (median ~2e-2 on unseen params). Box was retrospectively ~25% chaotic → Lyapunov-filtered evaluation added |
| 8 | `step08` (nb 5, Part B) | Same on a verified chaos-free box (0/48 draws chaotic) | Clean verdicts: interpolation tie; family-level force laws recovered; in-box error spikes diagnosed as **transient phase slips / mirror-twin basin calls**, not fit failures |
| 9 | `step09` (nb 5, Part C) | Does 4.5× more data (9M pairs) fix Step 8's imperfections? | Fixes fit quality (additive becomes the clear interpolation winner; spikes shrink 2–5×; a wrong basin call flips correct). Does **not** move beyond-box extrapolation — that's architectural |
| 10 | `step10` (nb 5, Part D) | Ignore transients: does the model settle into the **same pattern** (period + orbit)? | Pattern fidelity survives 2–3× further in parameter space than pointwise accuracy. Failures localize to two known mechanisms (chaos-killing at low δ; d-saturation at high δ) |

## Headline scoreboard (Step 9: 128 param sets × 35 ICs = 8.96M pairs, 60k steps)

| Metric | Scalar (black box) | Additive (f + d + g) | Better |
|---|---|---|---|
| Parameters | 4,737 | 3,587 | additive |
| Final eval loss (MSE) | 2.28e-9 | 1.50e-9 | additive |
| Unseen params — median rollout err | 1.67e-2 | **1.30e-2** | additive |
| γ-ray median err | 4.5e-2 | **3.6e-2** | additive |
| δ-ray median err | **2.7e-2** | 6.3e-2 | scalar |
| ω-ray median err | 9.4e-3 | 9.4e-3 | tie |
| Settled pattern correct — γ (of 61) | 29 | **34** | additive |
| Settled pattern correct — δ (of 61) | **60** | 31 | scalar |
| Settled pattern correct — ω (of 61) | 47 | **50** | additive |

Interpretation: the additive model is the better *scientific instrument*
(interpolation at scale, interpretability — it hands back f = x − x³, d = −δẋ,
g = γcos ωt as inspectable curves and discovers ω is redundant); the scalar model is
more robust at the edges (δ-extrapolation, preserving near-onset chaos). Both scalar
wins trace to one additive disease — tanh subnets saturating along one input
direction — motivating the queued next rung: a polynomial/bilinear skeleton
(f = c₁x + c₃x³ + MLP, d = −δẋ·(1 + correction)).

## Reproducing

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd scripts
cp ../data/*.npz .          # weight caches — REQUIRED to avoid retraining
../.venv/bin/python step09_family_scaled.py
```

- Every training function caches to `params_<tag>.npz` in the working directory and
  prints "loaded cached params" when reusing; **delete a cache file to retrain that
  model from scratch**. With caches present, most scripts run in ~1–5 min (pure
  simulation/plotting); fresh retraining ranges from ~2 min (steps 1–4) to ~25 min
  (step 9, two 60k-step trainings on CPU).
- `step10_settled_pattern.py` trains nothing — it only needs
  `params_family3_{scalar,additive}.npz` beside it (~15 min of long simulations).
- Scripts show figures interactively (`plt.show()`); running the notebooks via
  jupyter instead reproduces everything in place.
- All randomness is seeded (data generation, inits, batch streams, LHS draws) —
  results are bit-reproducible on CPU.

## Method notes (common to all steps)

- Integrator: fixed-step RK4; data saved at dt = 0.01 s (internal dt = 0.005 s).
- Training: one-step state-prediction MSE through one RK4 step; Adam, batch 16,384,
  lr 1e-3 → 1e-4 step drop; identical init seed / batch stream across compared models.
- Long-run structure: exact strobing (save step locked to T_drive/SPP — a 0.01-grid
  strobe smears points by ~1e-2, enough to corrupt period detection), settled-period
  test max_k ‖s(k+m) − s(k)‖, Benettin two-trajectory Lyapunov estimates.
- Family experiments: Latin-hypercube parameter draws, per-oscillator IC boxes scaled
  to 80% of that oscillator's measured attractor extent, chaos screening at λ > 0.02/s.

There is a companion track (parameter-space atlas: 61k–4.4M-point classification
sweeps and interactive 3-D viewers) not included here — happy to send it as well.
