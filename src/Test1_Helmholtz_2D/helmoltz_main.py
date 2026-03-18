import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax.numpy as jnp
import jax
from jax import vmap, jit, jacfwd
import optax
from functools import partial
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import sys
import time
from datetime import datetime

sys.path.insert(0, '../')
from KAN import KAN


L_range = (-1.0, 1.0)
H_range = (-1.0, 1.0)
nx, ny = 200, 200
x_interior = np.linspace(L_range[0], L_range[1], nx)
y_interior = np.linspace(H_range[0], H_range[1], ny)

x_left   = np.array(L_range[0]).reshape(-1)
y_left   = y_interior
x_right  = np.array(L_range[1]).reshape(-1)
y_right  = y_interior
x_bottom = x_interior
y_bottom = np.array(H_range[0]).reshape(-1)
x_top    = x_interior
y_top    = np.array(H_range[1]).reshape(-1)


class SF_KAN_Separable:
    def __init__(self, layer_dims, init_lr, num_epochs, k=3, r=5, mode='spikan'):
        self.input_size = layer_dims[0]
        self.out_size   = layer_dims[-1]
        self.r          = r
        self.layer_dims = [self.input_size] + layer_dims[1:-1] + [self.r * self.out_size]

        self.model_x = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False,
                           add_bias=True, grid_e=0.02, j='0', mode=mode)
        self.model_y = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False,
                           add_bias=True, grid_e=0.02, j='0', mode=mode)

        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        self.variables_x = self.model_x.init(key1, jnp.ones([1, 1]))
        self.variables_y = self.model_y.init(key2, jnp.ones([1, 1]))

        scheduler = optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=num_epochs,
            alpha=0.001
        )
        self.optimizer   = optax.adam(learning_rate=scheduler, nesterov=True)
        self.opt_state_x = self.optimizer.init(self.variables_x['params'])
        self.opt_state_y = self.optimizer.init(self.variables_y['params'])
        self.train_losses = []

    def predict(self, x, y):
        preds, _ = self.forward_pass(self.variables_x, self.variables_y, x, y)
        return preds

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables_x, variables_y, x, y):
        preds_x, spl_regs_x = self.model_x.apply(variables_x, x[:, None])
        preds_y, spl_regs_y = self.model_y.apply(variables_y, y[:, None])
        preds_x = preds_x.reshape(-1, self.out_size, self.r)
        preds_y = preds_y.reshape(-1, self.out_size, self.r)
        preds   = jnp.einsum('ijk,ljk->ilj', preds_x, preds_y)
        return preds, spl_regs_x + spl_regs_y

    @partial(jit, static_argnums=(0,))
    def loss(self, params_x, params_y, state_x, state_y, *args):
        variables_x = {'params': params_x, 'state': state_x}
        variables_y = {'params': params_y, 'state': state_y}
        return self.loss_fn(variables_x, variables_y, *args)

    @partial(jit, static_argnums=(0,))
    def train_step(self, params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args):
        (loss_value, (physics_loss, boundary_loss)), grads = jax.value_and_grad(
            self.loss, has_aux=True, argnums=(0, 1)
        )(params_x, params_y, state_x, state_y, *args)
        grads_x, grads_y = grads

        updates_x, opt_state_x = self.optimizer.update(grads_x, opt_state_x)
        updates_y, opt_state_y = self.optimizer.update(grads_y, opt_state_y)
        params_x = optax.apply_updates(params_x, updates_x)
        params_y = optax.apply_updates(params_y, updates_y)

        return params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss

    def train(self, num_epochs, *args):
        params_x, state_x = self.variables_x['params'], self.variables_x['state']
        params_y, state_y = self.variables_y['params'], self.variables_y['state']
        opt_state_x, opt_state_y = self.opt_state_x, self.opt_state_y
        loss_history = []

        pbar = trange(num_epochs, smoothing=0.)
        for epoch in pbar:
            params_x, params_y, opt_state_x, opt_state_y, loss_value, physics_loss, boundary_loss = self.train_step(
                params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args
            )
            loss_history.append(loss_value)
            if epoch % 10 == 0:
                pbar.set_postfix({
                    'Total Loss':    f"{loss_value:.4e}",
                    'Physics Loss':  f"{physics_loss:.4e}",
                    'Boundary Loss': f"{boundary_loss:.4e}"
                })

        self.variables_x = {'params': params_x, 'state': state_x}
        self.variables_y = {'params': params_y, 'state': state_y}
        return loss_history


class Helmholtz_SF_KAN_Separable(SF_KAN_Separable):
    def __init__(self, *args, kappa=1.0, a1=1.0, a2=4.0, mode='spikan', **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.kappa = kappa
        self.a1    = a1
        self.a2    = a2

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, variables_x, variables_y, x_interior, y_interior,
                x_left, y_left, x_right, y_right, x_bottom, y_bottom, x_top, y_top):
        residuals    = self.compute_residuals(variables_x, variables_y, x_interior, y_interior)
        physics_loss = jnp.mean(jnp.square(residuals))

        preds_left,   _ = self.forward_pass(variables_x, variables_y, x_left,   y_left)
        preds_right,  _ = self.forward_pass(variables_x, variables_y, x_right,  y_right)
        preds_bottom, _ = self.forward_pass(variables_x, variables_y, x_bottom, y_bottom)
        preds_top,    _ = self.forward_pass(variables_x, variables_y, x_top,    y_top)

        boundary_loss = (jnp.mean(jnp.square(preds_left))   +
                         jnp.mean(jnp.square(preds_right))  +
                         jnp.mean(jnp.square(preds_bottom)) +
                         jnp.mean(jnp.square(preds_top)))
        return physics_loss + boundary_loss, (physics_loss, boundary_loss)

    @partial(jit, static_argnums=(0,))
    def compute_residuals(self, variables_x, variables_y, x_interior, y_interior):
        def model_x_func(x):
            return self.model_x.apply(variables_x, x.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        def model_y_func(y):
            return self.model_y.apply(variables_y, y.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        x_feats  = vmap(model_x_func)(x_interior)
        y_feats  = vmap(model_y_func)(y_interior)
        x_hess   = vmap(jacfwd(jacfwd(model_x_func)))(x_interior)
        y_hess   = vmap(jacfwd(jacfwd(model_y_func)))(y_interior)

        u_x       = x_feats[:, 0, :]
        u_y       = y_feats[:, 0, :]
        d2u_x_dx2 = x_hess[:, 0, :]
        d2u_y_dy2 = y_hess[:, 0, :]

        u       = jnp.einsum('ir,jr->ij', u_x, u_y)
        d2u_dx2 = jnp.einsum('ir,jr->ij', d2u_x_dx2, u_y)
        d2u_dy2 = jnp.einsum('ir,jr->ij', u_x, d2u_y_dy2)

        q = (-(self.a1 * jnp.pi) ** 2 * jnp.sin(self.a1 * jnp.pi * x_interior[:, None]) * jnp.sin(
            self.a2 * jnp.pi * y_interior[None, :])
             - (self.a2 * jnp.pi) ** 2 * jnp.sin(self.a1 * jnp.pi * x_interior[:, None]) * jnp.sin(
                    self.a2 * jnp.pi * y_interior[None, :])
             + self.kappa ** 2 * jnp.sin(self.a1 * jnp.pi * x_interior[:, None]) * jnp.sin(
                    self.a2 * jnp.pi * y_interior[None, :]))

        return d2u_dx2 + d2u_dy2 + self.kappa ** 2 * u - q



layer_dims = [1, 5, 5, 1]
init_lr    = 1e-3
k          = 3
r          = 5
kappa      = 1.0
a1         = 1.0
a2         = 4.0
num_epochs = 20000
nx_plot, ny_plot = 100, 100

timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir    = os.path.join('./data',    timestamp)
results_dir = os.path.join('./results', timestamp)
os.makedirs(save_dir,    exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

modes_to_test = [
    'spikan', 'or-spikan', 'xor-spikan',
    'tanh_or-spikan', 'tanh_xor-spikan',
    'sigmoid_or-spikan', 'sigmoid_xor-spikan'
]

times_per_iter = {}
l2_errors      = {}
max_errors     = {}
mean_errors    = {}
saved_datasets = {}

x_plot = np.linspace(L_range[0], L_range[1], nx_plot)
y_plot = np.linspace(H_range[0], H_range[1], ny_plot)
X_mesh, Y_mesh   = np.meshgrid(x_plot, y_plot)
u_exact_plot     = np.sin(a1 * np.pi * X_mesh) * np.sin(a2 * np.pi * Y_mesh)


for current_mode in modes_to_test:
    model = Helmholtz_SF_KAN_Separable(
        layer_dims=layer_dims,
        init_lr=init_lr,
        num_epochs=num_epochs,
        k=k, r=r,
        kappa=kappa, a1=a1, a2=a2,
        mode=current_mode,
    )

    start_time   = time.time()
    loss_history = model.train(
        num_epochs,
        x_interior, y_interior,
        x_left, y_left, x_right, y_right,
        x_bottom, y_bottom, x_top, y_top
    )
    end_time    = time.time()
    ms_per_iter = (end_time - start_time) / num_epochs * 1000
    times_per_iter[current_mode] = ms_per_iter

    u_pred = np.array(model.predict(x_plot, y_plot)).squeeze().T

    error   = np.abs(u_pred - u_exact_plot)
    l2_err  = np.sqrt(np.sum(error ** 2)) / np.sqrt(np.sum(u_exact_plot ** 2)) * 100
    max_ae  = np.max(error)
    mean_ae = np.mean(error)

    l2_errors[current_mode]   = {'u': l2_err}
    max_errors[current_mode]  = {'u': max_ae}
    mean_errors[current_mode] = {'u': mean_ae}

    saved_datasets[current_mode] = {'u': u_pred}

    output_data = {
        'mesh':            {'x_mesh': X_mesh, 'y_mesh': Y_mesh, 'L_range': L_range},
        'field_variables': {'u': u_pred},
        'parameters':      {'kappa': kappa, 'a1': a1, 'a2': a2},
        'training':        {'loss_history': loss_history},
        'ms_per_iter':     ms_per_iter
    }
    save_path = os.path.join(
        save_dir,
        f'2d_helmholtz_{current_mode}_kappa{kappa}_nx{nx}_ny{ny}_epochs{num_epochs}_{layer_dims}.npy'
    )
    np.save(save_path, output_data)




for mode in modes_to_test:
    print(f"[{mode}] Time: {times_per_iter[mode]:.3f} ms/iter")
    print(f"   - u: L2 {l2_errors[mode]['u']:.3f}%, Max AE {max_errors[mode]['u']:.4f}, Mean AE {mean_errors[mode]['u']:.4f}")

txt_save_path = os.path.join(results_dir, 'evaluation_summary.txt')
with open(txt_save_path, 'w', encoding='utf-8') as f:
    f.write("=== Hyperparameters ===\n")
    f.write(f"nx: {nx}\n")
    f.write(f"ny: {ny}\n")
    f.write(f"nx_plot: {nx_plot}\n")
    f.write(f"ny_plot: {ny_plot}\n")
    f.write(f"kappa: {kappa}\n")
    f.write(f"a1: {a1}, a2: {a2}\n")
    f.write(f"Layer dims: {layer_dims}\n")
    f.write(f"Learning rate (init_lr): {init_lr}\n")
    f.write(f"k: {k}\n")
    f.write(f"r: {r}\n")
    f.write(f"Num epochs: {num_epochs}\n\n")

    f.write("=== Evaluation Results ===\n")
    for mode in modes_to_test:
        f.write(f"Mode: {mode}\n")
        f.write(f"  - Time per iter: {times_per_iter[mode]:.3f} ms/iter\n")
        f.write(f"  [u]\n")
        f.write(f"    - L2 Error:       {l2_errors[mode]['u']:.3f} %\n")
        f.write(f"    - Max Abs Error:  {max_errors[mode]['u']:.4f}\n")
        f.write(f"    - Mean Abs Error: {mean_errors[mode]['u']:.4f}\n")
        f.write("-" * 40 + "\n")


plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 12,
})

def set_tick_format(ax, cbar, decimals=2):
    formatter = ticker.FormatStrFormatter(f'%.{decimals}f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    cbar.ax.yaxis.set_major_formatter(formatter)

u_exact_vis = np.sin(a1 * np.pi * X_mesh) * np.sin(a2 * np.pi * Y_mesh)

fig = plt.figure(figsize=(21, 18))
gs  = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.45,
                        left=0.08, right=0.92, top=0.95, bottom=0.05)

def plot_helper(row, col, u_val, title_base, is_error=False):
    ax   = plt.subplot(gs[row, col])
    cmap = 'RdBu_r'
    if is_error:
        title = f'Absolute Error,\n{title_base}'
        label = r'$|u_{\text{pred}} - u_{\text{exact}}|$'
        im    = ax.contourf(X_mesh, Y_mesh, u_val, levels=50, cmap=cmap)
    else:
        title = "Reference Solution" if title_base == "REF" else f'Predicted $u$,\n{title_base}'
        label = r'$u$'
        if title_base == "REF":
            im = ax.contourf(X_mesh, Y_mesh, u_val, levels=50, cmap=cmap, vmin=-1, vmax=1)
        else:
            im = ax.contourf(X_mesh, Y_mesh, u_val, levels=50, cmap=cmap)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    set_tick_format(ax, cbar, decimals=2)

plot_helper(0, 0, u_exact_vis, "REF")
plot_helper(1, 0, saved_datasets['spikan']['u'],    "SPIKAN")
plot_helper(1, 1, np.abs(saved_datasets['spikan']['u']    - u_exact_vis), "SPIKAN",    is_error=True)
plot_helper(2, 0, saved_datasets['or-spikan']['u'], "OR-SPIKAN")
plot_helper(2, 1, np.abs(saved_datasets['or-spikan']['u'] - u_exact_vis), "OR-SPIKAN", is_error=True)
plot_helper(3, 0, saved_datasets['xor-spikan']['u'],"XOR-SPIKAN")
plot_helper(3, 1, np.abs(saved_datasets['xor-spikan']['u']- u_exact_vis), "XOR-SPIKAN",is_error=True)

other_modes = ['tanh_or-spikan', 'tanh_xor-spikan', 'sigmoid_or-spikan', 'sigmoid_xor-spikan']
for idx, mode in enumerate(other_modes):
    u_p         = saved_datasets[mode]['u']
    title_upper = mode.replace('_', '\n').upper()
    plot_helper(idx, 2, u_p,                          title_upper)
    plot_helper(idx, 3, np.abs(u_p - u_exact_vis),   title_upper, is_error=True)

final_plot_path = os.path.join(results_dir, 'comparison_helmholtz2d_4x4_optimized.png')
plt.savefig(final_plot_path,                    dpi=600, bbox_inches='tight')
plt.savefig(final_plot_path.replace('.png', '.pdf'),     bbox_inches='tight')
plt.show()
