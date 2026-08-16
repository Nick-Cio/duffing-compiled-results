"""Step 1 — single-trajectory neural ODE

One oscillator (delta=0.5, gamma=0.5, omega=1.7), one trajectory from (0,0).
MLP 4->64->64->2 on inputs (x, xdot, cos wt, sin wt); one-step RK4 training.
Produces: rollout accuracy, vector-field heatmap, the wrong-well failure at (-0.5, 0).

Extracted verbatim from the executed notebook 1_duffing_node_foundations.ipynb (this script is the
exact code that produced the results). Training cells cache weights to
params_<tag>.npz beside the script; delete a cache to retrain that model.
"""


# # Duffing Neural ODE — Step 1: single trajectory
#
# Learn the vector field of the forced Duffing oscillator
#
# $$\ddot{x} + \delta \dot{x} - x + x^3 = \gamma \cos(\omega t)$$
#
# from data alone, at one fixed parameter point $\delta=0.5$, $\gamma=0.5$, $\omega=1.7$ (one-well periodic for every initial condition, per the regime atlas) and one fixed initial condition $(x_0, \dot{x}_0) = (0, 0)$ — balanced on top of the potential barrier.
#
# **Setup:** the network $f_\theta(x, \dot{x}, \cos\omega t, \sin\omega t) \to (\dot{x}, \ddot{x})$ plays the role of the right-hand side of the ODE. The forcing phase enters as $(\cos\omega t, \sin\omega t)$ rather than raw $t$ so that time never leaves the training distribution and long rollouts stay legitimate.
#
# **Training:** single-step — from each observed state, integrate $f_\theta$ one save-step forward with RK4 and match the next observed state.
#
# **Evaluation:** teacher-forced one-step error, free rollout vs truth, rollout past the training horizon, and a learned-vs-true vector field comparison.


import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

# System: x'' + delta x' - x + x^3 = gamma cos(omega t)
DELTA, GAMMA, OMEGA = 0.5, 0.5, 1.7
IC = jnp.array([0.0, 0.0])

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
    # integrate f from (s0, t0), saving every dt_save, with `substeps` RK4 steps per save
    dt = dt_save / substeps
    def sub(c, _):
        s, t = c
        return (rk4_step(f, s, t, dt), t + dt), None
    def save_step(carry, _):
        carry, _ = jax.lax.scan(sub, carry, None, length=substeps)
        return carry, carry[0]
    _, traj = jax.lax.scan(save_step, (s0, t0), None, length=n_save)
    return jnp.concatenate([s0[None], traj])

# Ground truth: 20 s, integrate dt = 0.005, save every 0.01 s
T_END, DT_INT, DT_SAVE = 20.0, 0.005, 0.01
SUBS = int(round(DT_SAVE / DT_INT))
N_SAVE = int(round(T_END / DT_SAVE))

times = np.arange(N_SAVE + 1) * DT_SAVE
data = np.asarray(simulate(true_rhs, IC, 0.0, N_SAVE, DT_SAVE, SUBS))
print('trajectory:', data.shape, ' final state:', data[-1])


# ## Training data
#
# Starting balanced on the barrier top, the trajectory falls into a well and spirals onto the limit cycle. The transient is the part that carries information about the vector field away from the attractor.


T_FORCE = 2 * np.pi / OMEGA          # forcing period
STROBE = np.round(np.arange(0, T_END, T_FORCE) / DT_SAVE).astype(int)

def dress_phase_axes(a):
    # atlas-gallery styling: gray x at both well bottoms, barrier line at x=0
    a.axvline(0, color='0.85', lw=0.8, zorder=0)
    a.plot([-1, 1], [0, 0], 'x', color='gray', ms=9, mew=2)
    a.set_xlabel('x'); a.set_ylabel(r'$\dot{x}$')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(times, data[:, 0], lw=0.9)
ax[0].set_xlabel('t'); ax[0].set_ylabel('x'); ax[0].set_title('training trajectory x(t)')
ax[1].plot(data[:, 0], data[:, 1], lw=0.7)
ax[1].plot(data[STROBE, 0], data[STROBE, 1], 'k.', ms=6)
ax[1].plot([0], [0], 'o', color='k', ms=6, mfc='none')
dress_phase_axes(ax[1])
ax[1].set_title('phase portrait (dots = Poincare strobe, o = IC, x = well bottoms)')
plt.tight_layout(); plt.show()


# ## Model
#
# A small MLP (4 → 64 → 64 → 2, tanh) is the learned right-hand side. tanh rather than ReLU: the function gets *integrated*, and smooth activations give smooth flows.


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

n_params = sum(W.size + b.size for W, b in init_params(jax.random.PRNGKey(0)))
print('parameters:', n_params)


# ## Training
#
# Single-step objective: for every consecutive pair $(s_n, s_{n+1})$ in the data, RK4-integrate $f_\theta$ one save-step forward from $s_n$ and take the MSE against $s_{n+1}$. All 2,000 pairs in one full batch, Adam with a simple learning-rate decay.


states = jnp.asarray(data)
t_grid = jnp.asarray(times)
s_now, s_next, t_now = states[:-1], states[1:], t_grid[:-1]

def one_step(params, s, t):
    return rk4_step(lambda s_, t_: node_rhs(params, s_, t_), s, t, DT_SAVE)

def loss_fn(params):
    pred = jax.vmap(lambda s, t: one_step(params, s, t))(s_now, t_now)
    return jnp.mean((pred - s_next) ** 2)

def adam_update(params, grads, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, v, grads)
    mhat = jax.tree.map(lambda m_: m_ / (1 - b1 ** step), m)
    vhat = jax.tree.map(lambda v_: v_ / (1 - b2 ** step), v)
    new = jax.tree.map(lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
                       params, mhat, vhat)
    return new, m, v

@jax.jit
def train_step(params, m, v, step, lr):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    params, m, v = adam_update(params, grads, m, v, step, lr)
    return params, m, v, loss

params = init_params(jax.random.PRNGKey(0))
m = jax.tree.map(jnp.zeros_like, params)
v = jax.tree.map(jnp.zeros_like, params)

N_STEPS = 6000
for i in range(1, N_STEPS + 1):
    if i <= 4000:
        lr = 1e-3
    else:
        lr = 1e-4
    params, m, v, loss = train_step(params, m, v, float(i), lr)
    if i == 1 or i % 500 == 0:
        print(f'step {i:5d}   lr {lr:.0e}   loss {loss:.3e}')


# ## Evaluation 1 — teacher-forced one-step error
#
# Prediction error one step ahead, starting from the *true* state each time. This only confirms the fit worked; it says nothing yet about rollout stability.


pred = np.asarray(jax.vmap(lambda s, t: one_step(params, s, t))(s_now, t_now))
err = np.linalg.norm(pred - np.asarray(s_next), axis=1)
print(f'one-step error: median {np.median(err):.2e}   max {err.max():.2e}')

plt.figure(figsize=(9, 3))
plt.semilogy(times[:-1], err, lw=0.8)
plt.xlabel('t'); plt.ylabel('|error| per step'); plt.title('teacher-forced one-step error')
plt.tight_layout(); plt.show()


# ## Evaluation 2 — free rollout
#
# The real exam: integrate the learned vector field from the IC for the full 20 s with **no access to the data**, and overlay on the truth. Errors compound step to step, so any weakness shows up as drift.


node_f = lambda s, t: node_rhs(params, s, t)
roll = np.asarray(simulate(node_f, IC, 0.0, N_SAVE, DT_SAVE, 1))
roll_err = np.linalg.norm(roll - data, axis=1)
print(f'rollout error: mean {roll_err.mean():.2e}   final {roll_err[-1]:.2e}')

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(times, data[:, 0], 'k', lw=1.6, label='true')
ax[0].plot(times, roll[:, 0], 'r--', lw=1.1, label='NODE rollout')
ax[0].set_xlabel('t'); ax[0].set_ylabel('x'); ax[0].legend()
ax[0].set_title('free rollout vs truth')
ax[1].plot(data[:, 0], data[:, 1], 'k', lw=0.8)
ax[1].plot(roll[:, 0], roll[:, 1], 'r--', lw=0.7)
ax[1].plot(roll[STROBE, 0], roll[STROBE, 1], 'k.', ms=6)
dress_phase_axes(ax[1])
ax[1].set_title('phase portrait (dots = rollout strobe)')
plt.tight_layout(); plt.show()


# ## Evaluation 3 — past the training horizon
#
# Rollout to 40 s — twice the training window. Because time enters only as $(\cos\omega t, \sin\omega t)$, $t > 20$ is *not* extrapolation for the network: the inputs stay on the same circle it trained on, so it should keep tracing the limit cycle indefinitely.


N_LONG = int(round(40.0 / DT_SAVE))
t_long = np.arange(N_LONG + 1) * DT_SAVE
true_long = np.asarray(simulate(true_rhs, IC, 0.0, N_LONG, DT_SAVE, SUBS))
roll_long = np.asarray(simulate(node_f, IC, 0.0, N_LONG, DT_SAVE, 1))
tail_err = np.linalg.norm(roll_long - true_long, axis=1)[t_long > T_END]
print(f'error beyond training horizon: mean {tail_err.mean():.2e}   max {tail_err.max():.2e}')

plt.figure(figsize=(11, 3.5))
plt.plot(t_long, true_long[:, 0], 'k', lw=1.6, label='true')
plt.plot(t_long, roll_long[:, 0], 'r--', lw=1.1, label='NODE rollout')
plt.axvline(T_END, color='gray', ls=':', lw=1.5)
plt.text(T_END + 0.3, plt.ylim()[1] * 0.85, 'end of training data', color='gray')
plt.xlabel('t'); plt.ylabel('x'); plt.legend(loc='lower right')
plt.title('rollout to 2x the training horizon')
plt.tight_layout(); plt.show()

# phase portrait of the settled part only (t in [20, 40]): the transient spiral is
# gone and what remains is the limit cycle — a single closed loop
late = t_long >= T_END
strobe_long = np.round(np.arange(0, 40.0, T_FORCE) / DT_SAVE).astype(int)
strobe_late = strobe_long[t_long[strobe_long] >= T_END]
fig, a = plt.subplots(figsize=(6.5, 4.5))
a.plot(true_long[late, 0], true_long[late, 1], 'k', lw=1.4, label='true')
a.plot(roll_long[late, 0], roll_long[late, 1], 'r--', lw=1.0, label='NODE rollout')
a.plot(roll_long[strobe_late, 0], roll_long[strobe_late, 1], 'k.', ms=8)
dress_phase_axes(a)
a.legend()
a.set_title('phase portrait, t = 20 to 40 s: the limit cycle\n'
            '(strobe dots collapse to one point = period-1)')
plt.tight_layout(); plt.show()


# ## Evaluation 4 — where is the learned vector field actually right?
#
# Compare the learned acceleration $f_\theta \to \ddot{x}$ against the true $-\delta\dot{x} + x - x^3 + \gamma\cos(\omega t)$ over the whole $(x, \dot{x})$ plane at forcing phase $t=0$. The white curve is the single training trajectory. Expect: accurate inside the tube the trajectory painted, wrong elsewhere — the network only knows the vector field where it has seen data. This is the picture that motivates Step 2 (many initial conditions = paint the whole region).


xs = np.linspace(-2.0, 2.0, 161)
vs = np.linspace(-2.0, 2.0, 161)
X, V = np.meshgrid(xs, vs)
grid = jnp.stack([jnp.asarray(X.ravel()), jnp.asarray(V.ravel())], axis=1)

learned_acc = np.asarray(jax.vmap(lambda s: node_rhs(params, s, 0.0))(grid))[:, 1]
learned_acc = learned_acc.reshape(X.shape)
true_acc = -DELTA * V + X - X**3 + GAMMA * np.cos(0.0)
abs_err = np.abs(learned_acc - true_acc)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
lim = np.abs(true_acc).max()
for a, Z, title, cmap, vmin, vmax in [
        (ax[0], true_acc, 'true acceleration', 'coolwarm', -lim, lim),
        (ax[1], learned_acc, 'learned acceleration', 'coolwarm', -lim, lim),
        (ax[2], abs_err, '|error|  (log color)', 'magma', None, None)]:
    if title.startswith('|'):
        im = a.pcolormesh(X, V, np.log10(Z + 1e-6), cmap=cmap)
    else:
        im = a.pcolormesh(X, V, Z, cmap=cmap, vmin=vmin, vmax=vmax)
    a.plot(data[:, 0], data[:, 1], 'w', lw=1.0, alpha=0.9)
    a.set_xlabel('x'); a.set_ylabel('dx/dt'); a.set_title(title)
    fig.colorbar(im, ax=a, shrink=0.85)
plt.tight_layout(); plt.show()


# ## Evaluation 5 — unseen initial conditions
#
# The network only saw one trajectory. Roll it out from ICs it never trained on, spanning near-to-far from the training tube:
#
# - **(0.8, 0.3)** — inside the tube the training spiral painted: should still work
# - **(0.0, 0.8)** — off the tube, but its transient passes near painted territory: partial credit at best
# - **(−0.5, 0.0)** — the true trajectory settles into the **left well**, which the network has never seen: expect nonsense


TEST_ICS = [(0.8, 0.3), (0.0, 0.8), (-0.5, 0.0)]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for col, ic in enumerate(TEST_ICS):
    s0 = jnp.array(ic)
    tru = np.asarray(simulate(true_rhs, s0, 0.0, N_SAVE, DT_SAVE, SUBS))
    rol = np.asarray(simulate(node_f, s0, 0.0, N_SAVE, DT_SAVE, 1))
    err = np.linalg.norm(rol - tru, axis=1)
    print(f'IC {ic}:  mean err {err.mean():.2e}   final err {err[-1]:.2e}')

    a = axes[0, col]
    a.plot(times, tru[:, 0], 'k', lw=1.4, label='true')
    a.plot(times, rol[:, 0], 'r--', lw=1.0, label='NODE rollout')
    a.set_xlabel('t'); a.set_ylabel('x')
    a.set_title(f'IC = {ic}')
    if col == 0:
        a.legend()

    a = axes[1, col]
    a.plot(data[:, 0], data[:, 1], color='0.75', lw=0.6, label='training data')
    a.plot(tru[:, 0], tru[:, 1], 'k', lw=1.0)
    a.plot(rol[:, 0], rol[:, 1], 'r--', lw=0.8)
    a.plot([ic[0]], [ic[1]], 'o', color='k', ms=6, mfc='none')
    dress_phase_axes(a)
    if col == 0:
        a.legend()
plt.tight_layout(); plt.show()
