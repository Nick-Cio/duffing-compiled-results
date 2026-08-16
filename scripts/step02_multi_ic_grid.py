"""Step 2 — 63-IC training grid

Same oscillator, 9x7 grid of initial conditions. Introduces the extrapolation-ray
methodology (sweep x0 with xdot0=0 and vice versa, mean 20 s rollout error).

Extracted verbatim from the executed notebook 1_duffing_node_foundations.ipynb (this script is the
exact code that produced the results). Training cells cache weights to
params_<tag>.npz beside the script; delete a cache to retrain that model.
"""


# # Duffing Neural ODE — Step 2: many initial conditions
#
# Same system and parameter point as Step 1 ($\delta=0.5$, $\gamma=0.5$, $\omega=1.7$), but now the network trains on trajectories from a **9×7 grid of initial conditions** covering $x_0 \in [-0.75, 0.75]$, $\dot{x}_0 \in [-0.4, 0.4]$ — 63 trajectories, 20 s each, straddling both wells.
#
# Goals:
# 1. **Interpolation** — held-out ICs inside the box (off the training grid) should now work, including $(-0.5, 0)$ which failed completely in Step 1.
# 2. **Extrapolation breakdown** — march ICs along rays out of the box and measure where the rollout error takes off.


import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

DELTA, GAMMA, OMEGA = 0.5, 0.5, 1.7

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

T_END, DT_INT, DT_SAVE = 20.0, 0.005, 0.01
SUBS = int(round(DT_SAVE / DT_INT))
N_SAVE = int(round(T_END / DT_SAVE))
times = np.arange(N_SAVE + 1) * DT_SAVE

# 9x7 grid of ICs over the training box
X0_LIM, V0_LIM = 0.75, 0.4
ic_grid = np.stack(np.meshgrid(np.linspace(-X0_LIM, X0_LIM, 9),
                               np.linspace(-V0_LIM, V0_LIM, 7),
                               indexing='ij'), axis=-1).reshape(-1, 2)
print('training ICs:', ic_grid.shape)

sim_batch = jax.vmap(lambda s0: simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))
trajs = np.asarray(sim_batch(jnp.asarray(ic_grid)))
print('trajectories:', trajs.shape)
n_right = int((trajs[:, -1, 0] > 0).sum())
print(f'settle to right well: {n_right}/63, left well: {63 - n_right}/63')


# ## Training data coverage
#
# All 63 trajectories in the phase plane. This is the "painted region" — the network can only be trusted where there is paint. Note the trajectories overshoot the IC box (black rectangle): coverage is set by where trajectories *go*, not where they start.


T_FORCE = 2 * np.pi / OMEGA

def dress_phase_axes(a):
    a.axvline(0, color='0.85', lw=0.8, zorder=0)
    a.plot([-1, 1], [0, 0], 'x', color='gray', ms=9, mew=2)
    a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')

fig, a = plt.subplots(figsize=(8, 6))
for tr in trajs:
    a.plot(tr[:, 0], tr[:, 1], color='0.7', lw=0.3, zorder=1)
a.plot(ic_grid[:, 0], ic_grid[:, 1], 'k.', ms=4, zorder=3)
a.add_patch(plt.Rectangle((-X0_LIM, -V0_LIM), 2 * X0_LIM, 2 * V0_LIM,
                          fill=False, edgecolor='k', lw=1.2, zorder=3))
dress_phase_axes(a)
a.set_title('63 training trajectories (dots = ICs, box = IC sampling range)')
plt.tight_layout(); plt.show()


# ## Model and training
#
# Identical architecture to Step 1 (4 → 64 → 64 → 2, tanh; inputs $x, \dot{x}, \cos\omega t, \sin\omega t$). The dataset is now ~126k step pairs, so training is minibatched: 16,384 random pairs per Adam step.


LAYERS = [4, 64, 64, 2]

def init_params(key):
    params = []
    for n_in, n_out in zip(LAYERS[:-1], LAYERS[1:]):
        key, k = jax.random.split(key)
        W = jax.random.normal(k, (n_in, n_out)) * jnp.sqrt(1.0 / n_in)
        params.append((W, jnp.zeros(n_out)))
    return params

def node_rhs(params, s, t):
    h = jnp.array([s[0], s[1], jnp.cos(OMEGA * t), jnp.sin(OMEGA * t)])
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return h @ W + b

# flatten all (state_n, state_n+1, t_n) pairs across trajectories
s_now = jnp.asarray(trajs[:, :-1, :].reshape(-1, 2))
s_next = jnp.asarray(trajs[:, 1:, :].reshape(-1, 2))
t_now = jnp.asarray(np.tile(times[:-1], len(ic_grid)))
N_PAIRS = s_now.shape[0]
print('training pairs:', N_PAIRS)

def one_step(params, s, t):
    return rk4_step(lambda s_, t_: node_rhs(params, s_, t_), s, t, DT_SAVE)

def loss_on(params, sb, snb, tb):
    pred = jax.vmap(lambda s, t: one_step(params, s, t))(sb, tb)
    return jnp.mean((pred - snb) ** 2)

def adam_update(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mhat = jax.tree.map(lambda m_: m_ / (1 - b1 ** step), m)
    vhat = jax.tree.map(lambda v_: v_ / (1 - b2 ** step), v)
    new = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                       params, mhat, vhat)
    return new, m, v

BATCH = 16384

@jax.jit
def train_step(params, m, v, step, lr, key):
    idx = jax.random.randint(key, (BATCH,), 0, N_PAIRS)
    loss, grads = jax.value_and_grad(loss_on)(params, s_now[idx], s_next[idx],
                                              t_now[idx])
    params, m, v = adam_update(params, grads, m, v, step, lr)
    return params, m, v, loss

params = init_params(jax.random.PRNGKey(0))
m = jax.tree.map(jnp.zeros_like, params)
v = jax.tree.map(jnp.zeros_like, params)
base_key = jax.random.PRNGKey(1)

N_STEPS = 12000
for i in range(1, N_STEPS + 1):
    if i <= 8000:
        lr = 1e-3
    else:
        lr = 1e-4
    params, m, v, loss = train_step(params, m, v, float(i), lr,
                                    jax.random.fold_in(base_key, i))
    if i == 1 or i % 1000 == 0:
        print(f'step {i:6d}   lr {lr:.0e}   batch loss {loss:.3e}')

full_loss = loss_on(params, s_now, s_next, t_now)
print(f'final loss over all {N_PAIRS} pairs: {full_loss:.3e}')


# ## Test 1 — interpolation: unseen ICs inside the box
#
# Five held-out ICs, none on the 9×7 training grid. The last one, $(-0.5, 0)$, is the IC that failed by ~500× in Step 1 (the network had never seen the left well). Now the left well is painted.


node_f = lambda s, t: node_rhs(params, s, t)

INTERP_ICS = [(0.33, 0.17), (-0.61, -0.29), (0.05, -0.37), (0.7, 0.35), (-0.5, 0.0)]
results = {}
for ic in INTERP_ICS:
    s0 = jnp.array(ic)
    tru = np.asarray(simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))
    rol = np.asarray(simulate(node_f, s0, 0.0, N_SAVE, DT_SAVE, 1))
    err = np.linalg.norm(rol - tru, axis=1)
    results[ic] = (tru, rol, err)
    same_well = (tru[-1, 0] > 0) == (rol[-1, 0] > 0)
    print(f'IC {str(ic):15s} mean err {err.mean():.2e}   final err {err[-1]:.2e}'
          f'   same final well: {same_well}')


SHOW = [(0.33, 0.17), (-0.61, -0.29), (-0.5, 0.0)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, ic in enumerate(SHOW):
    tru, rol, err = results[ic]
    a = axes[0, col]
    a.plot(times, tru[:, 0], 'k', lw=1.4, label='true')
    a.plot(times, rol[:, 0], 'r--', lw=1.0, label='NODE rollout')
    a.set_xlabel('t'); a.set_ylabel('x'); a.set_title(f'IC = {ic}')
    if col == 0:
        a.legend()
    a = axes[1, col]
    a.plot(tru[:, 0], tru[:, 1], 'k', lw=1.0)
    a.plot(rol[:, 0], rol[:, 1], 'r--', lw=0.8)
    a.plot([ic[0]], [ic[1]], 'o', color='k', ms=6, mfc='none')
    dress_phase_axes(a)
plt.tight_layout(); plt.show()


# ## Test 2 — where does extrapolation break down?
#
# March the IC along two rays out of the box and measure the mean rollout error at each starting point:
#
# - along $x_0$ with $\dot{x}_0 = 0$, from $-2.5$ to $2.5$ (box edge at $\pm 0.75$)
# - along $\dot{x}_0$ with $x_0 = 0$, from $-1.5$ to $1.5$ (box edge at $\pm 0.4$)
#
# Expect low flat error inside the box, staying low a bit *past* the edge (training trajectories overshoot the box, so the painted region is bigger), then a takeoff once the IC starts in truly unpainted territory. Spikes can also mark **basin boundaries**: there a tiny model error flips the trajectory into the wrong well, so the error jumps to order 1 even when the vector field is only slightly off.


def ray_errors(ics):
    ics = jnp.asarray(ics)
    tru = np.asarray(jax.vmap(
        lambda s0: simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))(ics))
    rol = np.asarray(jax.vmap(
        lambda s0: simulate(node_f, s0, 0.0, N_SAVE, DT_SAVE, 1))(ics))
    err = np.linalg.norm(rol - tru, axis=2).mean(axis=1)
    wrong_well = (tru[:, -1, 0] > 0) != (rol[:, -1, 0] > 0)
    return err, wrong_well

x_ray = np.linspace(-2.5, 2.5, 101)
v_ray = np.linspace(-1.5, 1.5, 61)
err_x, ww_x = ray_errors(np.stack([x_ray, np.zeros_like(x_ray)], axis=1))
err_v, ww_v = ray_errors(np.stack([np.zeros_like(v_ray), v_ray], axis=1))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for a, ray, err, ww, lim, name in [
        (ax[0], x_ray, err_x, ww_x, X0_LIM, '$x_0$  ($\\dot{x}_0 = 0$)'),
        (ax[1], v_ray, err_v, ww_v, V0_LIM, '$\\dot{x}_0$  ($x_0 = 0$)')]:
    a.semilogy(ray, err, 'k', lw=1.2)
    a.semilogy(ray[ww], err[ww], 'rv', ms=5, label='ends in wrong well')
    a.axvspan(-lim, lim, color='tab:blue', alpha=0.12, label='training IC box')
    a.set_xlabel(name); a.set_ylabel('mean rollout error')
    a.legend(loc='upper center', fontsize=9)
ax[0].set_title('extrapolation breakdown along $x_0$')
ax[1].set_title('extrapolation breakdown along $\\dot{x}_0$')
plt.tight_layout(); plt.show()
