import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
from tqdm import trange
import jax.numpy as jnp
import jax
from jax import vmap, jit, jacfwd
import optax
from functools import partial
from jax import random
import sys
import time
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
import matplotlib.ticker as ticker
from datetime import datetime
import os
sys.path.insert(0, '../')

from KAN import KAN


class AllenCahnSpectral1D:
    def __init__(self, Nx, D=1e-4):
        self.Nx = Nx
        self.D = D
        self.L = 2.0
        self.x_full = np.linspace(-1, 1, Nx)
        self.x = self.x_full[:-1]
        self.dx = self.L / (Nx - 1)
        self.kx = 2 * np.pi * fftfreq(Nx - 1)
        self.K2 = self.kx ** 2

    def set_initial_condition(self):
        return self.x ** 2 * np.cos(np.pi * self.x)

    def step_RK4(self, u, dt):
        k1 = dt * self.rhs(u)
        k2 = dt * self.rhs(u + 0.5 * k1)
        k3 = dt * self.rhs(u + 0.5 * k2)
        k4 = dt * self.rhs(u + k3)
        return u + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def rhs(self, u):
        u_hat = fft(u)
        d_hat = -self.D * self.K2 * u_hat
        n = 5.0 * (u - u ** 3)
        n_hat = fft(n)
        return ifft(d_hat + n_hat).real

    def solve(self, T_final, dt=1e-3):
        u = self.set_initial_condition()
        t = 0
        n_steps = int(np.ceil(T_final / dt))
        u_record = np.zeros((n_steps + 1, self.Nx))
        t_record = np.zeros(n_steps + 1)
        u_record[0] = np.append(u, u[0])
        t_record[0] = t
        pbar = trange(n_steps)
        for step in pbar:
            if t + dt > T_final:
                dt = T_final - t
            u = self.step_RK4(u, dt)
            t += dt
            u_record[step + 1] = np.append(u, u[0])
            t_record[step + 1] = t
            pbar.set_postfix({'t': f'{t:.3f}'})
        return u_record, t_record

    def get_grid(self):
        return self.x_full


Nx = 320
D = 1e-4
T_final = 1.0
dt = 1e-3

solver = AllenCahnSpectral1D(Nx=Nx, D=D)
u_record, t_record = solver.solve(T_final, dt=dt)
x_full = solver.get_grid()

T_new, X_new = np.meshgrid(t_record, x_full, indexing='ij')
U = u_record

plt.figure(figsize=(5, 8))
contour = plt.contourf(T_new, X_new, U, levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(contour, label='u(x, t)')
plt.xlabel('Time t')
plt.ylabel('x')
plt.xlim(0, 1.)
plt.ylim(-1, 1)
plt.tight_layout()
plt.show()


# ─── 도메인 설정 ─────────────────────────────────────────────
L_range = (-1.0, 1.0)
H_range = (0.0, 1.0)
nx, ny = 320, 160
x_interior = np.linspace(L_range[0], L_range[1], nx)
y_interior = np.linspace(H_range[0], H_range[1], ny)

x_left   = np.array([L_range[0]])
y_left   = y_interior
x_right  = np.array([L_range[1]])
y_right  = y_interior
x_bottom = x_interior
y_bottom = np.array([H_range[0]])


class SF_KAN_Separable:
    def __init__(self, layer_dims, init_lr, num_epochs, k=3, r=10, D=1e-4, mode='spikan'):
        self.input_size = layer_dims[0]
        self.out_size = layer_dims[-1]
        self.r = r
        self.layer_dims = [self.input_size] + layer_dims[1:-1] + [self.r * self.out_size]

        self.model_x = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, add_bias=True,
                           grid_e=0.02, j='0', mode=mode)
        self.model_y = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, add_bias=True,
                           grid_e=0.02, j='0', mode=mode)
        self.D = D

        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        self.variables_x = self.model_x.init(key1, jnp.ones([1, 1]))
        self.variables_y = self.model_y.init(key2, jnp.ones([1, 1]))

        scheduler = optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=num_epochs,
            alpha=0.001
        )
        self.optimizer = optax.adam(learning_rate=scheduler, nesterov=True)

        self.opt_state_x = self.optimizer.init(self.variables_x['params'])
        self.opt_state_y = self.optimizer.init(self.variables_y['params'])

        self.train_losses = []

    def interpolate_moments(self, mu_old, nu_old, new_shape):
        old_j = mu_old.shape[1]
        new_j = new_shape[1]
        old_indices = jnp.linspace(0, old_j - 1, old_j)
        new_indices = jnp.linspace(0, old_j - 1, new_j)
        interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)
        return vmap(interpolate_fn)(mu_old), vmap(interpolate_fn)(nu_old)

    def smooth_state_transition(self, old_state, params):
        adam_count = old_state[0].count
        adam_mu, adam_nu = old_state[0].mu, old_state[0].nu
        layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}
        for key in layer_keys:
            c_shape = params[key]['c_basis'].shape
            adam_mu[key]['c_basis'], adam_nu[key]['c_basis'] = self.interpolate_moments(
                adam_mu[key]['c_basis'], adam_nu[key]['c_basis'], c_shape)
        return (optax.ScaleByAdamState(adam_count, adam_mu, adam_nu), optax.ScaleByScheduleState(adam_count))

    @partial(jit, static_argnums=(0,))
    def predict(self, x, y):
        preds, _ = self.forward_pass(self.variables_x, self.variables_y, x, y)
        return preds

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables_x, variables_y, x, y):
        preds_x, spl_regs_x = self.model_x.apply(variables_x, x[:, None])
        preds_y, spl_regs_y = self.model_y.apply(variables_y, y[:, None])
        preds_x = preds_x.reshape(-1, self.out_size, self.r)
        preds_y = preds_y.reshape(-1, self.out_size, self.r)
        preds = jnp.einsum('ijk,ljk->ilj', preds_x, preds_y)
        return preds, spl_regs_x + spl_regs_y

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, variables_x, variables_y, x_interior, y_interior,
                x_left, y_left, x_right, y_right, x_bottom, y_bottom):
        residuals = self.compute_residuals(variables_x, variables_y, x_interior, y_interior)
        physics_loss = jnp.mean(jnp.square(residuals))

        preds_left, _   = self.forward_pass(variables_x, variables_y, x_left, y_left)
        preds_right, _  = self.forward_pass(variables_x, variables_y, x_right, y_right)
        preds_bottom, _ = self.forward_pass(variables_x, variables_y, x_bottom, y_bottom)

        left_loss    = jnp.mean(jnp.square(preds_left + 1))
        right_loss   = jnp.mean(jnp.square(preds_right + 1))
        bottom_exact = (x_bottom ** 2 * jnp.cos(jnp.pi * x_bottom))[:, None, None]
        bottom_loss  = jnp.mean(jnp.square(preds_bottom - bottom_exact))

        boundary_loss = left_loss + right_loss + bottom_loss
        return physics_loss + boundary_loss, (physics_loss, boundary_loss)

    @partial(jit, static_argnums=(0,))
    def compute_residuals(self, variables_x, variables_y, x_interior, y_interior):
        def model_x_func(x):
            return self.model_x.apply(variables_x, x.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        def model_y_func(y):
            return self.model_y.apply(variables_y, y.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        x_feats    = vmap(model_x_func)(x_interior)
        y_feats    = vmap(model_y_func)(y_interior)
        x_hess     = vmap(jacfwd(jacfwd(model_x_func)))(x_interior)
        y_grad     = vmap(jacfwd(model_y_func))(y_interior)

        u_x        = x_feats[:, 0, :]
        u_y        = y_feats[:, 0, :]
        d2u_x_dx2  = x_hess[:, 0, :]
        du_y_dy    = y_grad[:, 0, :]

        u       = jnp.einsum('ir,jr->ij', u_x, u_y)
        d2u_dx2 = jnp.einsum('ir,jr->ij', d2u_x_dx2, u_y)
        du_dy   = jnp.einsum('ir,jr->ij', u_x, du_y_dy)

        return du_dy - self.D * d2u_dx2 + 5 * (u ** 3 - u)

    @partial(jit, static_argnums=(0,))
    def loss(self, params_x, params_y, state_x, state_y, *args):
        return self.loss_fn(
            {'params': params_x, 'state': state_x},
            {'params': params_y, 'state': state_y},
            *args
        )

    @partial(jit, static_argnums=(0,))
    def train_step(self, params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args):
        (loss_value, (physics_loss, boundary_loss)), grads = jax.value_and_grad(
            self.loss, has_aux=True, argnums=(0, 1))(params_x, params_y, state_x, state_y, *args)
        grads_x, grads_y = grads
        updates_x, opt_state_x = self.optimizer.update(grads_x, opt_state_x)
        updates_y, opt_state_y = self.optimizer.update(grads_y, opt_state_y)
        params_x = optax.apply_updates(params_x, updates_x)
        params_y = optax.apply_updates(params_y, updates_y)
        return params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss

    def train(self, num_epochs, x_interior, y_interior, x_bottom, y_bottom, x_left, y_left, x_right, y_right):
        params_x, state_x = self.variables_x['params'], self.variables_x['state']
        params_y, state_y = self.variables_y['params'], self.variables_y['state']
        opt_state_x, opt_state_y = self.opt_state_x, self.opt_state_y
        loss_history = []

        pbar = trange(num_epochs, smoothing=0.)
        for epoch in pbar:
            params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss = self.train_step(
                params_x, params_y, state_x, state_y, opt_state_x, opt_state_y,
                x_interior, y_interior, x_left, y_left, x_right, y_right, x_bottom, y_bottom
            )
            loss_history.append(loss_value)
            if epoch % 10 == 0:
                pbar.set_postfix({
                    'Total Loss': f"{loss_value:.4e}",
                    'Physics Loss': f"{physics_loss:.4e}",
                    'Boundary Loss': f"{boundary_loss:.4e}"
                })

        self.variables_x = {'params': params_x, 'state': state_x}
        self.variables_y = {'params': params_y, 'state': state_y}
        return loss_history


layer_dims = [1, 5, 5, 1]
init_lr = 1e-3
k = 5
r = 20
num_epochs = 100000

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir   = os.path.join('./data', timestamp)
results_dir = os.path.join('./results', timestamp)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print(f"저장 폴더: {save_dir}")

ref_save_path = os.path.join(save_dir, 'allen_cahn_ref.npy')
np.save(ref_save_path, {'x': x_full, 't': t_record, 'u': u_record})
print(f"레퍼런스 저장 완료: {ref_save_path}")

modes_to_test = [
    'spikan',
    'or-spikan',
    'xor-spikan',
    'tanh_or-spikan',
    'tanh_xor-spikan',
    'sigmoid_or-spikan',
    'sigmoid_xor-spikan',
]

saved_paths = {}

for current_mode in modes_to_test:
    print(f"\n{'=' * 40}")
    print(f"[{current_mode}] 모드 학습 시작...")
    print(f"{'=' * 40}")

    model = SF_KAN_Separable(
        layer_dims=layer_dims, init_lr=init_lr, num_epochs=num_epochs,
        D=1e-4, k=k, r=r, mode=current_mode,
    )

    start_time = time.time()

    loss_history = model.train(
        num_epochs,
        x_interior, y_interior,
        x_bottom, y_bottom,
        x_left, y_left,
        x_right, y_right
    )

    end_time = time.time()
    ms_per_iter = (end_time - start_time) / num_epochs * 1000

    x_plot = np.linspace(-1, 1, nx)
    t_plot = np.linspace(0, 1, ny)
    u_pred = model.predict(x_plot, t_plot).squeeze().T

    output_data = {
        'x': x_plot, 't': t_plot, 'u': u_pred,
        'D': 1e-4, 'loss_history': loss_history,
        'ms_per_iter': ms_per_iter
    }

    save_path = os.path.join(
        save_dir,
        f'allen_cahn_{current_mode}_nx{nx}_nt{ny}_{layer_dims}_epochs{num_epochs}_k{k}_r{r}.npy'
    )
    np.save(save_path, output_data)
    saved_paths[current_mode] = save_path


def count_trainable_params(variables_x, variables_y):
    flat_x, _ = jax.tree_util.tree_flatten(variables_x['params'])
    flat_y, _ = jax.tree_util.tree_flatten(variables_y['params'])
    return sum(p.size for p in flat_x) + sum(p.size for p in flat_y)


plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 12,
})


def set_tick_format(ax, cbar, decimals=2):
    fmt = ticker.FormatStrFormatter(f'%.{decimals}f')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    cbar.ax.yaxis.set_major_formatter(fmt)


def interpolate_to_reference_grid(data, ref_t, ref_x):
    interp_func = RegularGridInterpolator(
        (np.array(data['t']), np.array(data['x'])),
        np.array(data['u']), method='linear', bounds_error=False, fill_value=None
    )
    T_ref, X_ref = np.meshgrid(ref_t, ref_x, indexing='ij')
    points = np.stack((T_ref.flatten(), X_ref.flatten()), axis=1)
    return interp_func(points).reshape(T_ref.shape)


spectral_data = np.load(ref_save_path, allow_pickle=True).item()
T_spectral, X_spectral = np.meshgrid(spectral_data['t'], spectral_data['x'], indexing='ij')

solutions = {
    mode: np.load(path, allow_pickle=True).item()
    for mode, path in saved_paths.items()
}

interpolated_solutions, errors, l2_errors = {}, {}, {}
max_errors, mean_errors, times_per_iter = {}, {}, {}

for name, data in solutions.items():
    u_interp = interpolate_to_reference_grid(data, spectral_data['t'], spectral_data['x'])
    interpolated_solutions[name] = u_interp
    abs_err = np.abs(u_interp - spectral_data['u'])
    errors[name] = abs_err

    l2_errors[name] = 100 * np.sqrt(np.mean(abs_err ** 2)) / np.sqrt(np.mean(spectral_data['u'] ** 2))
    max_errors[name] = np.max(abs_err)
    mean_errors[name] = np.mean(abs_err)
    times_per_iter[name] = data.get('ms_per_iter', 0.0)

    print(
        f"[{name}] L2 Error: {l2_errors[name]:.3f}% | Max AE: {max_errors[name]:.4f} | Mean AE: {mean_errors[name]:.4f} | Time: {times_per_iter[name]:.3f} ms/iter")

txt_save_path = os.path.join(results_dir, 'evaluation_summary.txt')
with open(txt_save_path, 'w', encoding='utf-8') as f:
    f.write("=== Hyperparameters ===\n")
    f.write(f"Nx (Spectral): {Nx}\n")
    f.write(f"nx (Model x-grid): {nx}\n")
    f.write(f"ny (Model t-grid): {ny}\n")
    f.write(f"D (Diffusion Coeff): {D}\n")
    f.write(f"Layer dims: {layer_dims}\n")
    f.write(f"Learning rate (init_lr): {init_lr}\n")
    f.write(f"k (Spline order): {k}\n")
    f.write(f"r (Rank): {r}\n")
    f.write(f"Num epochs: {num_epochs}\n\n")

    f.write("=== Evaluation Results ===\n")
    for name in solutions.keys():
        f.write(f"Mode: {name}\n")
        f.write(f"  - L2 Error:       {l2_errors[name]:.3f} %\n")
        f.write(f"  - Max Abs Error:  {max_errors[name]:.4f}\n")
        f.write(f"  - Mean Abs Error: {mean_errors[name]:.4f}\n")
        f.write(f"  - Time per iter:  {times_per_iter[name]:.3f} ms/iter\n")
        f.write("-" * 40 + "\n")


n_modes = len(solutions)
fig = plt.figure(figsize=(12, 4 * n_modes + 2))
gs = gridspec.GridSpec(n_modes, 3, width_ratios=[0.7, 1, 1], wspace=0.3, hspace=0.3)

ax_ref = plt.subplot(gs[:, 0])
im_ref = ax_ref.contourf(T_spectral, X_spectral, spectral_data['u'],
                         levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
cbar_ref = plt.colorbar(im_ref, ax=ax_ref, fraction=0.07, pad=0.04)
cbar_ref.set_label(r'$u$')
ax_ref.set_title('Reference Solution')
ax_ref.set_aspect('equal')
ax_ref.set_xlabel(r'$t$')
ax_ref.set_ylabel(r'$x$')
set_tick_format(ax_ref, cbar_ref)

for idx, (name, u_interp) in enumerate(interpolated_solutions.items()):
    ax_pred = plt.subplot(gs[idx, 1])
    im_pred = ax_pred.contourf(T_spectral, X_spectral, u_interp,
                               levels=50, cmap='RdBu_r', vmin=-1, vmax=1)
    cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
    cbar_pred.set_label(r'$u$')
    ax_pred.set_title(f'Predicted $u$, {name}')
    ax_pred.set_aspect('equal')
    ax_pred.set_xlabel(r'$t$')
    ax_pred.set_ylabel(r'$x$')
    set_tick_format(ax_pred, cbar_pred)

    ax_err = plt.subplot(gs[idx, 2])
    im_err = ax_err.contourf(T_spectral, X_spectral, errors[name],
                             levels=50, cmap='RdBu_r')
    cbar_err = plt.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)
    cbar_err.set_label(r'$|u_{\text{pred}} - u_{\text{exact}}|$')
    ax_err.set_title(f'Absolute Error, {name}  (L2={l2_errors[name]:.3f}%)')
    ax_err.set_aspect('equal')
    ax_err.set_xlabel(r'$t$')
    ax_err.set_ylabel(r'$x$')
    set_tick_format(ax_err, cbar_err)

plt.savefig(os.path.join(results_dir, 'comparison_allencahn.png'), dpi=600, bbox_inches='tight')
plt.show()