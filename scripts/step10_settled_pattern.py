"""Step 10 — settled-pattern analysis (Step-9 weights)

Ignores transients: detects each system's settled period (exact strobe over 448 drive
periods) and compares settled orbits by Hausdorff distance (mirror twin allowed) at
every ray point. Verdicts: same / mirror / period mismatch / shape mismatch /
went aperiodic / chaos killed / both aperiodic.
REQUIRES params_family3_scalar.npz and params_family3_additive.npz beside the script.

Extracted verbatim from the executed notebook 5_duffing_node_family_series.ipynb (this script is the
exact code that produced the results). Training cells cache weights to
params_<tag>.npz beside the script; delete a cache to retrain that model.
"""


# ## Settled-pattern analysis: does it land on the right orbit?
#
# The rollout-error rays punish early transient mistakes. This section asks the
# question that matters for long-run fidelity instead: **after the transient dies,
# does the learned system settle into the *same pattern* as the truth — same period,
# same orbit shape?**
#
# Protocol, for every point on the three parameter rays (Step-9 weights, IC $(0.5,0)$):
#
# 1. run truth and both models for 448 drive periods (~1000–2000 s);
# 2. exact-strobe the last 48 periods and detect the **settled period**: the smallest
#    $m \le 16$ with $\max_k\|s(k{+}m) - s(k)\| <$ tol (none → aperiodic);
# 3. compare settled orbits by symmetric Hausdorff distance between the last-16-period
#    curves — computed both directly and against the **mirror twin** ($x, \dot{x} \to
#    -x, -\dot{x}$), since landing on the symmetry partner is dynamically right too.
#
# Verdicts: **same** (period matches, orbits overlap), **mirror** (period matches, overlaps
# the twin), **period mismatch**, **shape mismatch**, **went aperiodic** (truth periodic,
# model not), **chaos killed** (truth aperiodic, model periodic), **both aperiodic**
# (outside the box the ray crosses real chaos — neither can nor should be periodic).


import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree

jax.config.update('jax_enable_x64', True)

P_LO = np.array([0.7, 3.5, 1.9])
P_HI = np.array([1.0, 4.0, 2.9])
CENTER = {'delta': 0.85, 'gamma': 3.75, 'omega': 2.4}

def true_rhs(s, t, ph):
    x, v = s
    delta, gamma, omega = ph
    return jnp.array([v, -delta * v + x - x**3 + gamma * jnp.cos(omega * t)])

def rk4_step(f, s, t, dt):
    k1 = f(s, t)
    k2 = f(s + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(s + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(s + dt * k3, t + dt)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def init_like(z):
    n_layers = sum(1 for k in z.files if k.startswith('W'))
    return [(jnp.asarray(z[f'W{i}']), jnp.asarray(z[f'b{i}']))
            for i in range(n_layers)]

def mlp(params, u):
    h = u
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b

params_s = init_like(np.load('params_family3_scalar.npz'))
params_a = init_like(np.load('params_family3_additive.npz'))
N_F = N_D = 3

def rhs_scalar(s, t, ph):
    delta, gamma, omega = ph
    u = jnp.array([s[0], s[1], jnp.cos(omega * t), jnp.sin(omega * t),
                   delta, gamma, omega])
    return jnp.array([s[1], mlp(params_s, u)[0]])

def rhs_additive(s, t, ph):
    delta, gamma, omega = ph
    f = mlp(params_a[:N_F], jnp.array([s[0]]))[0]
    d = mlp(params_a[N_F:N_F + N_D], jnp.array([s[1], delta]))[0]
    g = mlp(params_a[N_F + N_D:], jnp.array([jnp.cos(omega * t),
                                             jnp.sin(omega * t),
                                             gamma, omega]))[0]
    return jnp.array([s[1], f + d + g])

SPP = 128            # exact strobe: save step locked to T_drive / SPP
N_SETTLE, N_MEAS = 400, 48
N_PER = N_SETTLE + N_MEAS
IC = jnp.array([0.5, 0.0])

def settle_run(rhs_fn, substeps):
    def one(ph):
        omega = ph[2]
        dt_save = (2 * jnp.pi / omega) / SPP
        dt = dt_save / substeps
        def sub(c, _):
            s, t = c
            return (rk4_step(lambda s_, t_: rhs_fn(s_, t_, ph), s, t, dt),
                    t + dt), None
        def save_step(carry, _):
            carry, _ = jax.lax.scan(sub, carry, None, length=substeps)
            return carry, carry[0]
        _, traj = jax.lax.scan(save_step, (IC, 0.0), None, length=N_PER * SPP)
        return traj
    return jax.jit(jax.vmap(one))

run_true = settle_run(true_rhs, 4)
run_scalar = settle_run(rhs_scalar, 2)
run_add = settle_run(rhs_additive, 2)

P_TOL = 3e-3         # strobe-residual tolerance for period detection
H_TOL = 5e-2         # Hausdorff tolerance for "same orbit"

def detect_period(sp):
    for m in range(1, 17):
        if np.linalg.norm(sp[m:] - sp[:-m], axis=1).max() < P_TOL:
            return m
    return 0

def hausdorff(A, B):
    ta, tb = cKDTree(A), cKDTree(B)
    return max(tb.query(A)[0].max(), ta.query(B)[0].max())

def analyze(traj):
    sp = traj[::SPP][-N_MEAS:]
    curve = traj[-16 * SPP:]
    return detect_period(sp), curve

def verdict(pt, ct, pm, cm):
    dh = min(hausdorff(ct, cm), hausdorff(ct, -cm))
    if pt == 0 and pm == 0:
        return 'both aperiodic', dh
    if pt == 0:
        return 'chaos killed', dh
    if pm == 0:
        return 'went aperiodic', dh
    if pm != pt:
        return 'period mismatch', dh
    if hausdorff(ct, cm) < H_TOL:
        return 'same', dh
    if hausdorff(ct, -cm) < H_TOL:
        return 'mirror', dh
    return 'shape mismatch', dh


RAYS = [
    ('gamma', 1, np.linspace(2.0, 5.5, 61)),
    ('delta', 0, np.linspace(0.3, 1.7, 61)),
    ('omega', 2, np.linspace(1.2, 3.8, 61)),
]
COLORS = {'same': 'tab:green', 'mirror': 'tab:olive',
          'period mismatch': 'tab:orange', 'shape mismatch': 'tab:red',
          'went aperiodic': 'tab:red', 'chaos killed': 'tab:purple',
          'both aperiodic': '0.7'}
MARKS = {'same': 'o', 'mirror': 'D', 'period mismatch': 's',
         'shape mismatch': 'x', 'went aperiodic': 'x',
         'chaos killed': '*', 'both aperiodic': '.'}

results = {}
for pname, j, vals in RAYS:
    phs = np.tile([CENTER['delta'], CENTER['gamma'], CENTER['omega']],
                  (len(vals), 1))
    phs[:, j] = vals
    phj = jnp.asarray(phs)
    tr_t = np.asarray(run_true(phj))
    tr_s = np.asarray(run_scalar(phj))
    tr_a = np.asarray(run_add(phj))
    rows = {}
    for tag, tr in [('scalar', tr_s), ('additive', tr_a)]:
        vs, ds, ps = [], [], []
        for k in range(len(vals)):
            pt, ct = analyze(tr_t[k])
            pm, cm = analyze(tr[k])
            v, dh = verdict(pt, ct, pm, cm)
            vs.append(v); ds.append(dh); ps.append((pt, pm))
        rows[tag] = (vs, np.array(ds), ps)
    results[pname] = (vals, j, rows)
    for tag in ('scalar', 'additive'):
        vs = rows[tag][0]
        n_ok = sum(v in ('same', 'mirror', 'both aperiodic') for v in vs)
        print(f'{pname:6s} {tag:9s} pattern-correct {n_ok}/{len(vals)}   '
              + '  '.join(f'{v}:{vs.count(v)}' for v in COLORS if v in vs))


fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))
for a, (pname, jdx, vals) in zip(ax, RAYS):
    _, j, rows = results[pname]
    for tag, y0, in [('scalar', 1.0), ('additive', 0.0)]:
        vs, ds, ps = rows[tag]
        base = {'scalar': 'tab:blue', 'additive': 'tab:green'}[tag]
        a.semilogy(vals, np.maximum(ds, 1e-4), color=base, lw=1.0,
                   alpha=0.45, label=f'{tag}: orbit distance')
    ymin, ymax = a.get_ylim()
    row_y = {'scalar': ymin * 2.2, 'additive': ymin * 1.3}
    for tag in ('scalar', 'additive'):
        vs, ds, ps = rows[tag]
        for v in COLORS:
            idx = [k for k, vv in enumerate(vs) if vv == v]
            if idx:
                a.plot(vals[idx], [row_y[tag]] * len(idx), MARKS[v],
                       color=COLORS[v], ms=5, ls='none')
    a.axvline(P_LO[j], color='tab:red', lw=1.2, ls='--')
    a.axvline(P_HI[j], color='tab:red', lw=1.2, ls='--')
    a.set_xlabel(pname)
    a.set_title(f'settled pattern along {pname}\n'
                'marker rows: top = scalar, bottom = additive')
    a.set_ylim(ymin, ymax)
a.legend(fontsize=7, loc='upper right')
ax[0].set_ylabel('settled-orbit Hausdorff distance (mirror allowed)')

from matplotlib.lines import Line2D
handles = [Line2D([], [], marker=MARKS[v], color=COLORS[v], ls='none',
                  label=v) for v in COLORS]
fig.legend(handles=handles, loc='lower center', ncol=7, fontsize=8,
           frameon=False)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.show()


# **How to read it:** the faint curves are the distance between the settled orbits
# (truth vs model, mirror allowed) — down at $10^{-2\text{--}3}$ the two limit cycles are
# visually identical. The marker rows are the verdicts per ray point (top row scalar,
# bottom additive). Anything green/olive/grey is a *pattern success*: the model settles
# into the exact true pattern, its mirror twin, or correctly stays aperiodic where the
# truth is. Orange/red/purple are genuine structural failures — and their locations
# (relative to the red training-range lines) say whether structure survives further in
# parameter space than pointwise accuracy does.
