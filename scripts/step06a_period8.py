"""Step 6a — period-8 regime

delta=0.7, gamma=3.175, omega=2.425: freshly period-doubled orbit (8 strobe points
in 4 pairs ~0.02 apart). Exact-strobe periodicity tables applied to the LEARNED models.

Extracted verbatim from the executed notebook 4_duffing_node_regimes.ipynb (this script is the
exact code that produced the results). Training cells cache weights to
params_<tag>.npz beside the script; delete a cache to retrain that model.
"""


# # Duffing Neural ODE — Step 6a: the period-8 cross-well regime
#
# $\delta = 0.7$, $\gamma = 3.175$, $\omega = 2.425$. Pre-characterization on our
# integrator (reproduced in the tour below):
#
# - a **genuine period-8 orbit**: the strobe returns to itself after exactly 8 drive
#   periods (residual $\sim 5\times10^{-9}$), and never sooner;
# - the 8 strobe points sit in **4 tight pairs** only $\approx 0.02$ apart — this regime
#   is freshly period-doubled from period-4, so a learned model must resolve
#   0.02-scale structure in the strobe map to reproduce the true period instead of
#   collapsing back to period-4;
# - cross-well, $|x| \lesssim 1.46$, $|\dot{x}| \lesssim 1.69$; contraction is weak
#   ($\lambda \approx -0.02\,/\mathrm{s}$, e-fold time $\sim 60$ s), so transients are
#   long and every 20 s training trajectory is mostly transient — fine for one-step
#   vector-field learning, but the long-run tests below use exact strobing after a long
#   settle;
# - odd symmetry gives a **mirror-twin** orbit ($x \to -x$, $\dot{x} \to -\dot{x}$,
#   $t \to t + T/2$); some ICs land on one twin, some on the other.
#
# Same two models and training process as Step 5 (scalar black box vs additive
# $f_\theta(x) + d_\theta(\dot{x}) + g_\theta(\cos\omega t, \sin\omega t)$, both with
# $\dot{x}_1 = x_2$ hardwired). The new headline metric: apply the strobe periodicity
# test **to the learned models** — do they hold the period-8 structure, or do they lock
# onto the wrong rung of the doubling cascade?


import time

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

DELTA, GAMMA, OMEGA = 0.7, 3.175, 2.425
T_DRIVE = 2 * np.pi / OMEGA
PERIOD_N = 8
ATTRACTOR_IC = (0.5, 0.0)
SECOND_IC = None
TAG = 'p8'

def true_rhs(s, t):
    x, v = s
    return jnp.array([v, -DELTA * v + x - x**3 + GAMMA * jnp.cos(OMEGA * t)])

def rk4_step(f, s, t, dt):
    k1 = f(s, t)
    k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(s + dt * k3, t + dt)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(f, s0, t0, n_save, dt_save, substeps):
    dt = dt_save / substeps
    def sub(c, _):
        s, t = c
        return (rk4_step(f, s, t, dt), t + dt), None
    def save_step(carry, _):
        carry, _ = jax.lax.scan(sub, carry, None, length=substeps)
        return carry, carry[0]
    _, traj = jax.lax.scan(save_step, (s0, t0), None, length=n_save)
    return jnp.concatenate([s0[None], traj])

DT_INT, DT_SAVE = 0.005, 0.01
SUBS = int(round(DT_SAVE / DT_INT))

# exact-strobe machinery: save step locked to T_DRIVE/SPP so strobe samples land
# exactly on drive-period multiples (a 0.01-grid strobe smears points by ~1e-2,
# which is the same size as the structure we need to resolve)
SPP = 64
DT_STROBE = T_DRIVE / SPP

def strobe_run(f, s0, n_per, substeps):
    traj = np.asarray(simulate(f, jnp.array(s0), 0.0, n_per * SPP,
                               DT_STROBE, substeps))
    return traj, traj[::SPP]

def period_table(sp, ms=(1, 2, 3, 4, 6, 8, 12)):
    for m in ms:
        d = np.linalg.norm(sp[m:] - sp[:-m], axis=1).max()
        marker = '  <-- period' if d < 1e-6 else ''
        print(f'   m={m:2d}   max |s(k+m) - s(k)| = {d:.2e}{marker}')


# ## 0 — Regime tour
#
# Dense trajectory, steady-state phase portrait (one full $n T$ orbit segment), and the
# exact strobe section with the periodicity table: the residual
# $\max_k \|s(k{+}m) - s(k)\|$ over late strobes drops to integrator precision exactly at
# the true period multiple.


N_SETTLE, N_MEAS = 400, 80
tour_full, tour_sp = strobe_run(true_rhs, ATTRACTOR_IC,
                                N_SETTLE + N_MEAS, 4)
steady_tr = tour_full[N_SETTLE * SPP:]
sp_true = tour_sp[N_SETTLE:]
orbit = tour_full[N_SETTLE * SPP:(N_SETTLE + PERIOD_N) * SPP + 1]

EXT_X = float(np.abs(steady_tr[:, 0]).max())
EXT_V = float(np.abs(steady_tr[:, 1]).max())
print(f'attractor extent:  |x| <= {EXT_X:.3f}   |xdot| <= {EXT_V:.3f}')
print(f'strobe periodicity from IC {ATTRACTOR_IC} '
      f'(last {N_MEAS} of {N_SETTLE + N_MEAS} periods):')
period_table(sp_true)

if SECOND_IC is not None:
    chaos_full, chaos_sp_all = strobe_run(true_rhs, SECOND_IC,
                                          N_SETTLE + N_MEAS * 4, 4)
    chaos_sp = chaos_sp_all[N_SETTLE:]
    print(f'strobe periodicity from IC {SECOND_IC}:')
    period_table(chaos_sp)

t_dense = np.arange(len(tour_full)) * DT_STROBE

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
a.plot(t_dense, tour_full[:, 0], lw=0.7)
a.axhline(1, color='0.8', lw=0.8)
a.axhline(-1, color='0.8', lw=0.8)
a.set_xlim(0, 25 * T_DRIVE)
a.set_xlabel('t'); a.set_ylabel('x')
a.set_title('x(t) — transient onto the orbit')

a = ax[1]
a.plot(orbit[:, 0], orbit[:, 1], 'k', lw=1.0)
a.plot([-1, 1], [0, 0], 'x', color='tab:red', ms=8, mew=2)
a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')
a.set_title(f'one full period-{PERIOD_N} orbit ({PERIOD_N} drive periods)')

a = ax[2]
a.plot(orbit[:, 0], orbit[:, 1], color='0.85', lw=0.6, zorder=0)
a.plot(sp_true[:, 0], sp_true[:, 1], 'o', color='tab:red', ms=6,
       label=f'period-{PERIOD_N} strobe')
if SECOND_IC is not None:
    a.plot(chaos_sp[:, 0], chaos_sp[:, 1], '.', color='0.4', ms=2.5,
           label='chaotic-set strobe (coexisting)')
    a.legend(fontsize=8)
a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')
a.set_title('exact strobe section')
plt.tight_layout(); plt.show()


# ## 1 — Training data, auto-scaled to the attractor
#
# Same recipe as Step 5: 9×7 IC grid at 80% of the measured extent, 20 s per
# trajectory. Contraction is weak here, so these trajectories are mostly transient —
# which is exactly what one-step vector-field training wants: broad phase-space
# coverage, not one settled orbit.


T_END = 20.0
N_SAVE = int(round(T_END / DT_SAVE))
times = np.arange(N_SAVE + 1) * DT_SAVE

X0_LIM = round(0.8 * EXT_X, 1)
V0_LIM = round(0.8 * EXT_V, 1)
print(f'IC box: x0 in [-{X0_LIM}, {X0_LIM}]   xdot0 in [-{V0_LIM}, {V0_LIM}]')

ic_grid = np.stack(np.meshgrid(np.linspace(-X0_LIM, X0_LIM, 9),
                               np.linspace(-V0_LIM, V0_LIM, 7),
                               indexing='ij'), axis=-1).reshape(-1, 2)

sim_batch = jax.vmap(lambda s0: simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))
trajs = np.asarray(sim_batch(jnp.asarray(ic_grid)))
print('training trajectories:', trajs.shape)


from scipy.ndimage import gaussian_filter
from matplotlib.patches import Rectangle, Patch

pts = trajs[:, :-1, :].reshape(-1, 2)
BIN, SIG = 0.02, 0.08
XB = np.arange(-1.6 * EXT_X, 1.6 * EXT_X + 1e-9, BIN)
VB = np.arange(-1.6 * EXT_V, 1.6 * EXT_V + 1e-9, BIN)
H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[XB, VB])
D_log = np.log10(1.0 + gaussian_filter(H, SIG / BIN))
D_norm = D_log / D_log.max()
xc = 0.5 * (XB[:-1] + XB[1:])
vc = 0.5 * (VB[:-1] + VB[1:])

def coverage_at(x, v):
    i = np.clip(np.searchsorted(xc, x), 0, len(xc) - 1)
    j = np.clip(np.searchsorted(vc, v), 0, len(vc) - 1)
    return D_norm[i, j]

fig, a = plt.subplots(figsize=(9, 6))
pc = a.pcolormesh(XB, VB, D_norm.T, cmap='Blues', vmin=0, vmax=1,
                  rasterized=True)
a.add_patch(Rectangle((-X0_LIM, -V0_LIM), 2 * X0_LIM, 2 * V0_LIM,
                      fill=False, edgecolor='tab:red', ls='--', lw=1.5))
a.plot([-1, 1], [0, 0], 'x', color='tab:red', ms=8, mew=2)
a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')
a.set_title('training-data coverage (darker = more data);'
            ' red dashed = IC box')
fig.colorbar(pc, ax=a, label='normalized log density')
plt.tight_layout(); plt.show()


# ## 2 — The two models


def init_mlp(key, layers):
    params = []
    for n_in, n_out in zip(layers[:-1], layers[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(1.0 / n_in)
        params.append((W, jnp.zeros(n_out)))
    return params

def mlp(params, u):
    h = u
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b

def make_input(s, t):
    return jnp.array([s[0], s[1], jnp.cos(OMEGA * t), jnp.sin(OMEGA * t)])

LAYERS_SCALAR = [4, 64, 64, 1]
def init_scalar(key):
    return init_mlp(key, LAYERS_SCALAR)
def rhs_scalar(params, s, t):
    a = mlp(params, make_input(s, t))[0]
    return jnp.array([s[1], a])

F_LAYERS = [1, 32, 32, 1]
D_LAYERS = [1, 32, 32, 1]
G_LAYERS = [2, 32, 32, 1]
N_F, N_D = len(F_LAYERS) - 1, len(D_LAYERS) - 1

def init_additive(key):
    kf, kd, kg = jax.random.split(key, 3)
    return (init_mlp(kf, F_LAYERS) + init_mlp(kd, D_LAYERS)
            + init_mlp(kg, G_LAYERS))

def additive_terms(params, s, t):
    pf = params[:N_F]
    pd = params[N_F:N_F + N_D]
    pg = params[N_F + N_D:]
    f = mlp(pf, jnp.array([s[0]]))[0]
    d = mlp(pd, jnp.array([s[1]]))[0]
    g = mlp(pg, jnp.array([jnp.cos(OMEGA * t), jnp.sin(OMEGA * t)]))[0]
    return f, d, g

def rhs_additive(params, s, t):
    f, d, g = additive_terms(params, s, t)
    return jnp.array([s[1], f + d + g])

for name, init in [('scalar', init_scalar), ('additive', init_additive)]:
    n = sum(W.size + b.size for W, b in init(jax.random.PRNGKey(0)))
    print(f'{name:9s} parameters: {n}')


# ## 3 — Training
#
# Identical harness to Step 5; caches `params_<tag>_<model>.npz` (delete to
# retrain).


import os

s_now = jnp.asarray(trajs[:, :-1, :].reshape(-1, 2))
s_next = jnp.asarray(trajs[:, 1:, :].reshape(-1, 2))
t_now = jnp.asarray(np.tile(times[:-1], len(ic_grid)))
N_PAIRS = s_now.shape[0]
print('training pairs:', N_PAIRS)

def adam_update(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mhat = jax.tree.map(lambda m_: m_ / (1 - b1 ** step), m)
    vhat = jax.tree.map(lambda v_: v_ / (1 - b2 ** step), v)
    new = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                       params, mhat, vhat)
    return new, m, v

BATCH = 16384
N_STEPS = 12000
EVAL_EVERY = 250

def train(rhs, init_fn, model_tag):
    def one_step(params, s, t):
        return rk4_step(lambda s_, t_: rhs(params, s_, t_), s, t, DT_SAVE)

    def loss_on(params, sb, snb, tb):
        pred = jax.vmap(lambda s, t: one_step(params, s, t))(sb, tb)
        return jnp.mean((pred - snb) ** 2)

    full_loss = jax.jit(lambda p: loss_on(p, s_now, s_next, t_now))

    cache = f'params_{TAG}_{model_tag}.npz'
    if os.path.exists(cache):
        z = np.load(cache)
        n_layers = sum(1 for k in z.files if k.startswith('W'))
        params = [(jnp.asarray(z[f'W{i}']), jnp.asarray(z[f'b{i}']))
                  for i in range(n_layers)]
        print(f'[{model_tag}] loaded cached params from {cache}   '
              f'full loss {float(full_loss(params)):.3e}   '
              f'(delete the file to retrain)')
        return (params, z['hist_steps'], z['hist_loss'], full_loss,
                float(z['wall']))

    @jax.jit
    def train_step(params, m, v, step, lr, key):
        idx = jax.random.randint(key, (BATCH,), 0, N_PAIRS)
        loss, grads = jax.value_and_grad(loss_on)(params, s_now[idx],
                                                  s_next[idx], t_now[idx])
        params, m, v = adam_update(params, grads, m, v, step, lr)
        return params, m, v, loss

    params = init_fn(jax.random.PRNGKey(0))
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    base_key = jax.random.PRNGKey(1)

    hist_steps, hist_loss = [], []
    t0 = time.time()
    for i in range(1, N_STEPS + 1):
        lr = 1e-3 if i <= 8000 else 1e-4
        params, m, v, _ = train_step(params, m, v, float(i), lr,
                                     jax.random.fold_in(base_key, i))
        if i == 1 or i % EVAL_EVERY == 0:
            fl = float(full_loss(params))
            hist_steps.append(i)
            hist_loss.append(fl)
            if i == 1 or i % 2000 == 0:
                print(f'[{model_tag}] step {i:6d}   lr {lr:.0e}'
                      f'   full loss {fl:.3e}')
    wall = time.time() - t0
    print(f'[{model_tag}] final full loss {hist_loss[-1]:.3e}'
          f'   wall time {wall:.0f} s')
    np.savez(cache, hist_steps=np.array(hist_steps),
             hist_loss=np.array(hist_loss), wall=wall,
             **{f'W{i}': np.asarray(Wt) for i, (Wt, _) in enumerate(params)},
             **{f'b{i}': np.asarray(bt) for i, (_, bt) in enumerate(params)})
    return params, np.array(hist_steps), np.array(hist_loss), full_loss, wall

params_s, steps_s, loss_s, _, wall_s = train(rhs_scalar, init_scalar, 'scalar')
params_a, steps_a, loss_a, _, wall_a = train(rhs_additive, init_additive,
                                             'additive')

MODELS = [('scalar', rhs_scalar, params_s, 'tab:blue'),
          ('additive', rhs_additive, params_a, 'tab:green')]


# ## 4 — Learning curves


fig, a = plt.subplots(figsize=(8, 5))
for (tag, _, _, color), st, lo in [(MODELS[0], steps_s, loss_s),
                                   (MODELS[1], steps_a, loss_a)]:
    a.semilogy(st, lo, color=color, lw=1.2, label=tag)
a.axvline(8000, color='0.85', lw=0.8, zorder=0)
a.set_xlabel('Adam step')
a.set_ylabel('full-dataset loss (MSE)')
a.set_title('learning curves — identical data, init seed, batches, schedule')
a.legend()
plt.tight_layout(); plt.show()

print(f'final loss   scalar {loss_s[-1]:.3e}   additive {loss_a[-1]:.3e}')
print(f'wall time    scalar {wall_s:.0f} s   additive {wall_a:.0f} s')


# ## 5 — Accuracy: one-step errors and held-out ICs
#
# Free 20 s rollouts from the five held-out interior ICs (Step-5 fractions of the new
# box). "Locked" = mean error over the final 5 s below 1.


def one_step_errors(rhs, params):
    f = lambda s, t: rk4_step(lambda s_, t_: rhs(params, s_, t_), s, t, DT_SAVE)
    pred = jax.vmap(f)(s_now, t_now)
    return np.asarray(jnp.linalg.norm(pred - s_next, axis=1))

for tag, rhs, p, _ in MODELS:
    e = one_step_errors(rhs, p)
    print(f'{tag:9s} one-step error   median {np.median(e):.2e}'
          f'   mean {e.mean():.2e}   max {e.max():.2e}')

FRACS = [(0.44, 0.42), (-0.81, -0.72), (0.07, -0.92), (0.93, 0.87),
         (-0.67, 0.0)]
INTERP_ICS = [(round(fx * X0_LIM, 2), round(fv * V0_LIM, 2))
              for fx, fv in FRACS]

LOCK_ERR = 1.0
lock_win = times[:-1] >= T_END - 5.0

def rollout_err(rhs, params, ic):
    s0 = jnp.array(ic)
    tru = np.asarray(simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))
    rol = np.asarray(simulate(lambda s, t: rhs(params, s, t),
                              s0, 0.0, N_SAVE, DT_SAVE, 1))
    err = np.linalg.norm(rol - tru, axis=1)
    locked = err[:-1][lock_win].mean() < LOCK_ERR
    return err.mean(), locked

print()
print('held-out IC rollouts (mean error over 20 s):')
print(f'{"IC":>16s} {"scalar":>12s} {"additive":>12s}   locked?')
for ic in INTERP_ICS:
    row, locks = [], []
    for tag, rhs, p, _ in MODELS:
        e, l = rollout_err(rhs, p, ic)
        row.append(e)
        locks.append(l)
    print(f'{str(ic):>16s} {row[0]:>12.2e} {row[1]:>12.2e}'
          f'   {locks[0]} / {locks[1]}')


# ## 6 — Extrapolation rays, with data coverage
#
# Rays to 1.5× the attractor extent along $x_0$ and 2.5× along $\dot{x}_0$, coverage
# strips behind, ▼ = lost phase lock by the end of the rollout.


def ray_errors(rhs, params, ics):
    ics = jnp.asarray(ics)
    f = lambda s, t: rhs(params, s, t)
    tru = np.asarray(jax.vmap(
        lambda s0: simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))(ics))
    rol = np.asarray(jax.vmap(
        lambda s0: simulate(f, s0, 0.0, N_SAVE, DT_SAVE, 1))(ics))
    err = np.linalg.norm(rol - tru, axis=2)
    lost = err[:, :-1][:, lock_win].mean(axis=1) > LOCK_ERR
    return err.mean(axis=1), lost

x_ray = np.linspace(-1.5 * EXT_X, 1.5 * EXT_X, 101)
v_ray = np.linspace(-2.5 * EXT_V, 2.5 * EXT_V, 101)
rays = {'x': np.stack([x_ray, np.zeros_like(x_ray)], axis=1),
        'v': np.stack([np.zeros_like(v_ray), v_ray], axis=1)}
res = {tag: {k: ray_errors(rhs, p, ics) for k, ics in rays.items()}
       for tag, rhs, p, _ in MODELS}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, keyk, ray, lim, name in [
        (ax[0], 'x', x_ray, X0_LIM, '$x_0$  ($\\dot{x}_0 = 0$)'),
        (ax[1], 'v', v_ray, V0_LIM, '$\\dot{x}_0$  ($x_0 = 0$)')]:
    cov = coverage_at(rays[keyk][:, 0], rays[keyk][:, 1])
    dr = ray[1] - ray[0]
    for xi, ci in zip(ray, cov):
        a.axvspan(xi - dr / 2, xi + dr / 2,
                  color=plt.cm.Blues(0.75 * ci), lw=0, zorder=0)
    for tag, _, _, color in MODELS:
        err, lost = res[tag][keyk]
        a.semilogy(ray, err, color=color, lw=1.2, label=tag)
        a.semilogy(ray[lost], err[lost], 'v', color=color, ms=5)
    a.axvline(-lim, color='tab:red', lw=1.2, ls='--')
    a.axvline(lim, color='tab:red', lw=1.2, ls='--', label='training IC box')
    a.set_xlabel(name); a.set_ylabel('mean rollout error')
    handles, _ = a.get_legend_handles_labels()
    handles.append(Patch(facecolor=plt.cm.Blues(0.6),
                         label='data coverage along ray'))
    a.legend(handles=handles, loc='upper center', fontsize=8)
ax[0].set_title('extrapolation along $x_0$ (▼ = lost phase lock)')
ax[1].set_title('extrapolation along $\\dot{x}_0$')
plt.tight_layout(); plt.show()

for keyk, ray in [('x', x_ray), ('v', v_ray)]:
    print(f'--- ray along {keyk}0 ---')
    for tag, *_ in MODELS:
        err, lost = res[tag][keyk]
        print(f'  {tag:9s} median err {np.median(err):.2e}'
              f'   lost-lock starts: {int(lost.sum())}/{len(ray)}')


# ## 7 — The long-run test: does the learned model keep the true period?
#
# Run each learned model from `ATTRACTOR_IC` for the same long settle as the truth,
# then apply the exact-strobe periodicity table to the **learned** trajectory. The
# right panel overlays orbits and strobe points (truth in black, its mirror twin in
# grey — a model is also "right" if it lands on the twin).


_, mir_sp_all = strobe_run(true_rhs, (-ATTRACTOR_IC[0], -ATTRACTOR_IC[1]),
                           N_SETTLE + N_MEAS, 4)
mir_full, _ = strobe_run(true_rhs, (-ATTRACTOR_IC[0], -ATTRACTOR_IC[1]),
                         N_SETTLE + PERIOD_N, 4)
mir_orbit = mir_full[N_SETTLE * SPP:]
mir_sp = mir_sp_all[N_SETTLE:]

fig, a = plt.subplots(figsize=(8, 6.5))
a.plot(orbit[:, 0], orbit[:, 1], 'k', lw=2.0, label='true orbit')
a.plot(mir_orbit[:, 0], mir_orbit[:, 1], color='0.75', lw=1.2,
       label='mirror twin')
a.plot(sp_true[:, 0], sp_true[:, 1], 'o', color='k', ms=9, mfc='none',
       mew=1.8)
for tag, rhs, p, color in MODELS:
    lfull, lsp_all = strobe_run(lambda s, t: rhs(p, s, t), ATTRACTOR_IC,
                                N_SETTLE + N_MEAS, 1)
    lorbit = lfull[N_SETTLE * SPP:(N_SETTLE + PERIOD_N) * SPP + 1]
    lsp = lsp_all[N_SETTLE:]
    print(f'learned [{tag}] strobe periodicity from IC {ATTRACTOR_IC}:')
    period_table(lsp)
    a.plot(lorbit[:, 0], lorbit[:, 1], '--', color=color, lw=1.2,
           label=f'{tag} orbit')
    a.plot(lsp[:, 0], lsp[:, 1], '.', color=color, ms=7)
a.plot([-1, 1], [0, 0], 'x', color='tab:red', ms=8, mew=2)
a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')
a.set_title(f'learned long-run orbits vs true period-{PERIOD_N}'
            ' (open circles = true strobe)')
a.legend(fontsize=8)
plt.tight_layout(); plt.show()


# ## 8 — The learned forces, term by term
#
# Gauge-fixed as in Steps 4–5: $g$ shifted to zero mean over the drive cycle, $d$
# shifted so $d(0) = 0$, both shifts absorbed into $f$.


term_f = jax.vmap(lambda x: additive_terms(params_a,
                  jnp.array([x, 0.0]), 0.0)[0])
term_d = jax.vmap(lambda v: additive_terms(params_a,
                  jnp.array([0.0, v]), 0.0)[1])

t_cycle = np.linspace(0.0, T_DRIVE, 200)
term_g = jax.vmap(lambda t: additive_terms(params_a,
                  jnp.array([0.0, 0.0]), t)[2])

g_vals = np.asarray(term_g(jnp.asarray(t_cycle)))
c_g = g_vals.mean()
c_d = float(term_d(jnp.array([0.0]))[0])
print(f'gauge constants:  mean(g) = {c_g:+.4f}   d(0) = {c_d:+.4f}'
      f'   (moved into f)')

x_plot = np.linspace(-1.5 * EXT_X, 1.5 * EXT_X, 400)
v_plot = np.linspace(-2.5 * EXT_V, 2.5 * EXT_V, 400)
f_vals = np.asarray(term_f(jnp.asarray(x_plot))) + c_g + c_d
d_vals = np.asarray(term_d(jnp.asarray(v_plot))) - c_d
g_tilde = g_vals - c_g

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
a = ax[0]
a.plot(x_plot, x_plot - x_plot**3, 'k', lw=1.8, label='true  $x - x^3$')
a.plot(x_plot, f_vals, '--', color='tab:green', lw=1.5,
       label=r'learned  $f_\theta(x)$')
a.axvspan(pts[:, 0].min(), pts[:, 0].max(), color='tab:blue', alpha=0.10,
          label='data range')
a.set_xlabel('x'); a.set_title('restoring force')
a.legend(fontsize=8)

a = ax[1]
a.plot(v_plot, -DELTA * v_plot, 'k', lw=1.8,
       label=r'true  $-\delta\dot{x}$')
a.plot(v_plot, d_vals, '--', color='tab:green', lw=1.5,
       label=r'learned  $d_\theta(\dot{x})$')
a.axvspan(pts[:, 1].min(), pts[:, 1].max(), color='tab:blue', alpha=0.10,
          label='data range')
a.set_xlabel(r'$\dot{x}$'); a.set_title('damping force')
a.legend(fontsize=8)

a = ax[2]
a.plot(t_cycle, GAMMA * np.cos(OMEGA * t_cycle), 'k', lw=1.8,
       label=r'true  $\gamma\cos\omega t$')
a.plot(t_cycle, g_tilde, '--', color='tab:green', lw=1.5,
       label=r'learned  $g_\theta$')
a.set_xlabel('t (one drive period)'); a.set_title('driving force')
a.legend(fontsize=8)
plt.tight_layout(); plt.show()

in_x = (x_plot > pts[:, 0].min()) & (x_plot < pts[:, 0].max())
in_v = (v_plot > pts[:, 1].min()) & (v_plot < pts[:, 1].max())
f_err = np.abs(f_vals - (x_plot - x_plot**3))
print(f'max |f error|   in data range {f_err[in_x].max():.2e}'
      f'   at edges of plot: {f_err[[0, -1]].max():.2e}')
print(f'max |d error|   in data range {np.abs(d_vals + DELTA * v_plot)[in_v].max():.2e}')
print(f'max |g error|   {np.abs(g_tilde - GAMMA * np.cos(OMEGA * t_cycle)).max():.2e}')


# ## What this rung settles
#
# Read off the results above:
#
# 1. **Rollout accuracy** transfers from Step 5 or doesn't — same box-scaled data
#    recipe, same budgets, weaker contraction to forgive mistakes.
# 2. **The learned strobe table is the sharp test**: period-8 means the model's own
#    strobe residual should dip at $m = 8$ (and only partially at $m = 4$ — the pair
#    splitting is real structure, size $\approx 0.02$). A model that is right to
#    $\mathcal{O}(10^{-2})$ in the vector field can still land on the wrong rung of the
#    cascade, because the doubling gap *is* $\mathcal{O}(10^{-2})$.
# 3. **The forces panel** says whether the additive decomposition stays exact when the
#    attractor is more delicate than the force laws themselves.
#
# Companion notebook: `duffing_node_p3.ipynb` — the period-3 regime, where a chaotic
# attractor coexists with the periodic one.
