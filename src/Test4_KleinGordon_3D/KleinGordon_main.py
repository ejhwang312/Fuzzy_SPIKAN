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



def set_tick_format(ax, cbar=None, decimals=2):
    formatter = ticker.FormatStrFormatter(f'%.{decimals}f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(formatter)


def exact_solution_np(x, y, t):
    return (x + y) * np.cos(t) + x * y * np.sin(t)



def create_interior_points(L_range, H_range, T_range, nx, ny, nt):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)
    t = np.linspace(T_range[0], T_range[1], nt)
    return x, y, t


def create_boundary_points(L_range, H_range, nx, ny):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)

    x_left   = jnp.array([L_range[0]])
    y_left   = y
    x_top    = x
    y_top    = jnp.array([H_range[1]])
    x_right  = jnp.array([L_range[1]])
    y_right  = y
    x_bottom = x
    y_bottom = jnp.array([H_range[0]])

    return (x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom)


L_range = (0., 1.0)
H_range = (0., 1.0)
T_range = (0.0, 10.0)
nx, ny, nt = 100, 100, 100

x_interior, y_interior, t_interior = create_interior_points(L_range, H_range, T_range, nx, ny, nt)

x_ic = x_interior
y_ic = y_interior
t_ic = jnp.array([0.])

(x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom) = create_boundary_points(
    L_range, H_range, nx, ny
)
t_bc = t_interior


class SF_KAN_Separable:
    def __init__(self, layer_dims, init_lr, num_epochs, k=3, r=10, mode='spikan'):
        self.input_size = layer_dims[0]
        self.out_size   = layer_dims[-1]
        self.r          = r
        self.layer_dims = [self.input_size] + layer_dims[1:-1] + [self.r * self.out_size]

        self.model_x = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False,
                           add_bias=True, grid_e=0.02, j='0', mode=mode)
        self.model_y = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False,
                           add_bias=True, grid_e=0.02, j='0', mode=mode)
        self.model_t = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False,
                           add_bias=True, grid_e=0.02, j='0', mode=mode)

        key1, key2, key3 = jax.random.split(jax.random.PRNGKey(10), 3)
        self.variables_x = self.model_x.init(key1, jnp.ones([1, 1]))
        self.variables_y = self.model_y.init(key2, jnp.ones([1, 1]))
        self.variables_t = self.model_t.init(key3, jnp.ones([1, 1]))

        scheduler = optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=num_epochs,
            alpha=0.001
        )
        self.optimizer   = optax.adam(learning_rate=scheduler, nesterov=True)
        self.opt_state_x = self.optimizer.init(self.variables_x['params'])
        self.opt_state_y = self.optimizer.init(self.variables_y['params'])
        self.opt_state_t = self.optimizer.init(self.variables_t['params'])

    def predict(self, x, y, t):
        preds, _ = self.forward_pass(self.variables_x, self.variables_y, self.variables_t, x, y, t)
        return preds

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables_x, variables_y, variables_t, x, y, t):
        preds_x, spl_regs_x = self.model_x.apply(variables_x, x[:, None])
        preds_y, spl_regs_y = self.model_y.apply(variables_y, y[:, None])
        preds_t, spl_regs_t = self.model_t.apply(variables_t, t[:, None])

        preds_x = preds_x.reshape(-1, self.out_size, self.r)
        preds_y = preds_y.reshape(-1, self.out_size, self.r)
        preds_t = preds_t.reshape(-1, self.out_size, self.r)

        preds    = jnp.einsum('ijk,ljk,mjk->ilm', preds_x, preds_y, preds_t)
        spl_regs = spl_regs_x + spl_regs_y + spl_regs_t
        return preds, spl_regs

    @partial(jit, static_argnums=(0,))
    def loss(self, params_x, params_y, params_t, state_x, state_y, state_t, *args):
        variables_x = {'params': params_x, 'state': state_x}
        variables_y = {'params': params_y, 'state': state_y}
        variables_t = {'params': params_t, 'state': state_t}
        return self.loss_fn(variables_x, variables_y, variables_t, *args)

    @partial(jit, static_argnums=(0,))
    def train_step(self, params_x, params_y, params_t, state_x, state_y, state_t,
                   opt_state_x, opt_state_y, opt_state_t, *args):
        (loss_value, (physics_loss, boundary_loss)), grads = jax.value_and_grad(
            self.loss, has_aux=True, argnums=(0, 1, 2)
        )(params_x, params_y, params_t, state_x, state_y, state_t, *args)
        grads_x, grads_y, grads_t = grads

        updates_x, opt_state_x = self.optimizer.update(grads_x, opt_state_x)
        updates_y, opt_state_y = self.optimizer.update(grads_y, opt_state_y)
        updates_t, opt_state_t = self.optimizer.update(grads_t, opt_state_t)

        params_x = optax.apply_updates(params_x, updates_x)
        params_y = optax.apply_updates(params_y, updates_y)
        params_t = optax.apply_updates(params_t, updates_t)

        return (params_x, params_y, params_t,
                opt_state_x, opt_state_y, opt_state_t,
                loss_value, physics_loss, boundary_loss)

    def train(self, num_epochs, *args):
        params_x, state_x = self.variables_x['params'], self.variables_x['state']
        params_y, state_y = self.variables_y['params'], self.variables_y['state']
        params_t, state_t = self.variables_t['params'], self.variables_t['state']
        opt_state_x, opt_state_y, opt_state_t = self.opt_state_x, self.opt_state_y, self.opt_state_t
        loss_history = []

        pbar = trange(num_epochs, smoothing=0.)
        for epoch in pbar:
            (params_x, params_y, params_t,
             opt_state_x, opt_state_y, opt_state_t,
             loss_value, physics_loss, boundary_loss) = self.train_step(
                params_x, params_y, params_t, state_x, state_y, state_t,
                opt_state_x, opt_state_y, opt_state_t, *args
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
        self.variables_t = {'params': params_t, 'state': state_t}
        return loss_history


class KleinGordon_SF_KAN_Separable(SF_KAN_Separable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exact_solution(self, x, y, t):
        return (x + y) * jnp.cos(t) + x * y * jnp.sin(t)

    def forcing_term(self, x, y, t):
        u = self.exact_solution(x, y, t)
        return u ** 2 - u

    @partial(jit, static_argnums=(0,))
    def compute_residuals(self, variables_x, variables_y, variables_t, x_interior, y_interior, t_interior):
        def model_x_func(x):
            return self.model_x.apply(variables_x, x.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        def model_y_func(y):
            return self.model_y.apply(variables_y, y.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        def model_t_func(t):
            return self.model_t.apply(variables_t, t.reshape(-1, 1))[0].reshape(self.out_size, self.r)

        x_feats = vmap(model_x_func)(x_interior)
        y_feats = vmap(model_y_func)(y_interior)
        t_feats = vmap(model_t_func)(t_interior)
        x_hess  = vmap(jacfwd(jacfwd(model_x_func)))(x_interior)
        y_hess  = vmap(jacfwd(jacfwd(model_y_func)))(y_interior)
        t_hess  = vmap(jacfwd(jacfwd(model_t_func)))(t_interior)

        u_x     = x_feats[:, 0, :]
        u_y     = y_feats[:, 0, :]
        u_t     = t_feats[:, 0, :]
        d2u_dx2 = x_hess[:, 0, :]
        d2u_dy2 = y_hess[:, 0, :]
        d2u_dt2 = t_hess[:, 0, :]

        u    = jnp.einsum('ir,jr,kr->ijk', u_x,     u_y,     u_t)
        u_tt = jnp.einsum('ir,jr,kr->ijk', u_x,     u_y,     d2u_dt2)
        u_xx = jnp.einsum('ir,jr,kr->ijk', d2u_dx2, u_y,     u_t)
        u_yy = jnp.einsum('ir,jr,kr->ijk', u_x,     d2u_dy2, u_t)

        f = self.forcing_term(
            x_interior[:, None, None],
            y_interior[None, :, None],
            t_interior[None, None, :]
        )
        return u_tt - (u_xx + u_yy) + u ** 2 - f

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, variables_x, variables_y, variables_t,
                x_interior, y_interior, t_interior,
                x_ic, y_ic, t_ic,
                x_left, y_left, x_top, y_top, x_right, y_right, x_bottom, y_bottom,
                t_bc):
        residuals    = self.compute_residuals(variables_x, variables_y, variables_t,
                                              x_interior, y_interior, t_interior)
        physics_loss = jnp.mean(jnp.square(residuals))

        ic_pred  = self.forward_pass(variables_x, variables_y, variables_t, x_ic, y_ic, t_ic)[0]
        ic_exact = self.exact_solution(x_ic[:, None, None], y_ic[None, :, None], t_ic[None, None, :])
        ic_loss  = jnp.mean(jnp.square(ic_pred - ic_exact))

        bc_left   = self.forward_pass(variables_x, variables_y, variables_t, x_left,   y_left,   t_bc)[0]
        bc_right  = self.forward_pass(variables_x, variables_y, variables_t, x_right,  y_right,  t_bc)[0]
        bc_bottom = self.forward_pass(variables_x, variables_y, variables_t, x_bottom, y_bottom, t_bc)[0]
        bc_top    = self.forward_pass(variables_x, variables_y, variables_t, x_top,    y_top,    t_bc)[0]

        bc_loss = (jnp.mean(jnp.square(bc_left   - self.exact_solution(x_left[:,None,None],   y_left[None,:,None],   t_bc[None,None,:]))) +
                   jnp.mean(jnp.square(bc_right   - self.exact_solution(x_right[:,None,None],  y_right[None,:,None],  t_bc[None,None,:]))) +
                   jnp.mean(jnp.square(bc_bottom  - self.exact_solution(x_bottom[:,None,None], y_bottom[None,:,None], t_bc[None,None,:]))) +
                   jnp.mean(jnp.square(bc_top     - self.exact_solution(x_top[:,None,None],    y_top[None,:,None],    t_bc[None,None,:]))))

        boundary_loss = ic_loss + bc_loss
        return physics_loss + boundary_loss, (physics_loss, boundary_loss)



layer_dims = [1, 5, 5, 1]
init_lr    = 1e-3
k          = 3
r          = 10
num_epochs = 50000

timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir    = os.path.join('./data',    timestamp)
results_dir = os.path.join('./results', timestamp)
os.makedirs(save_dir,    exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print(f"데이터 저장: {save_dir}")
print(f"결과 저장:   {results_dir}")

modes_to_test = [
    'spikan', 'or-spikan', 'xor-spikan',
    'tanh_or-spikan', 'tanh_xor-spikan',
    'sigmoid_or-spikan', 'sigmoid_xor-spikan'
]


X_ij, Y_ij = np.meshgrid(x_interior, y_interior, indexing='ij')  # (nx, ny)
u_exact_all = np.stack(
    [exact_solution_np(X_ij, Y_ij, t_val) for t_val in t_interior],
    axis=-1
)  # shape: (nx, ny, nt)
assert u_exact_all.shape == (nx, ny, nt), f"u_exact_all shape mismatch: {u_exact_all.shape}"

times_per_iter = {}
l2_errors      = {}
max_errors     = {}
mean_errors    = {}
saved_datasets = {}


for current_mode in modes_to_test:
    model = KleinGordon_SF_KAN_Separable(
        layer_dims=layer_dims,
        init_lr=init_lr,
        num_epochs=num_epochs,   # ★ 스케쥴러용
        k=k,
        r=r,
        mode=current_mode
    )

    start_time   = time.time()
    loss_history = model.train(
        num_epochs,
        x_interior, y_interior, t_interior,
        x_ic, y_ic, t_ic,
        x_left, y_left,
        x_top, y_top,
        x_right, y_right,
        x_bottom, y_bottom,
        t_bc
    )
    end_time    = time.time()
    ms_per_iter = (end_time - start_time) / num_epochs * 1000
    times_per_iter[current_mode] = ms_per_iter

    u_pred = np.array(model.predict(x_interior, y_interior, t_interior))
    assert u_pred.shape == (nx, ny, nt), f"[{current_mode}] u_pred shape mismatch: {u_pred.shape}"

    u_error = np.abs(u_pred - u_exact_all)
    l2_err  = np.sqrt(np.sum(u_error ** 2)) / np.sqrt(np.sum(u_exact_all ** 2)) * 100
    max_ae  = np.max(u_error)
    mean_ae = np.mean(u_error)

    l2_errors[current_mode]   = {'u': l2_err}
    max_errors[current_mode]  = {'u': max_ae}
    mean_errors[current_mode] = {'u': mean_ae}

    saved_datasets[current_mode] = u_pred


    output_data = {
        'mesh': {
            'nx': nx, 'ny': ny, 'nt': nt,
            'L_range': L_range, 'H_range': H_range, 'T_range': T_range
        },
        'field_variables': {
            'u_pred':  u_pred,
            'u_exact': u_exact_all,
            'u_error': u_error
        },
        'parameters': {
            'layer_dims': layer_dims, 'k': k, 'r': r
        },
        'training':   {'loss_history': loss_history},
        'ms_per_iter': ms_per_iter   # ★ 추가
    }

    save_path = os.path.join(
        save_dir,
        f'2d_kg_{current_mode}_nx{nx}_ny{ny}_nt{nt}_epochs{num_epochs}_{layer_dims}.npy'
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
    f.write(f"nt: {nt}\n")
    f.write(f"L_range: {L_range}\n")
    f.write(f"H_range: {H_range}\n")
    f.write(f"T_range: {T_range}\n")
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

t_idx_final = nt - 1
t_final_val = T_range[1]
u_exact_final = u_exact_all[:, :, t_idx_final]

fig = plt.figure(figsize=(21, 18))
gs  = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.45,
                        left=0.08, right=0.92, top=0.95, bottom=0.05)


def plot_helper_kg(row, col, u_val, title_base, is_error=False):
    ax   = plt.subplot(gs[row, col])
    cmap = 'RdBu_r'
    if is_error:
        title = f'Absolute Error,\n{title_base}'
        label = r'$|u_{\text{pred}} - u_{\text{exact}}|$'
    else:
        title = (f'Exact Solution\n(t = {t_final_val})' if title_base == "REF"
                 else f'{title_base}\n(t = {t_final_val})')
        label = r'$u$'

    im   = ax.contourf(X_ij, Y_ij, u_val, levels=50, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    set_tick_format(ax, cbar, decimals=2)


plot_helper_kg(0, 0, u_exact_final, "REF")
plot_helper_kg(1, 0, saved_datasets['spikan'][:, :, t_idx_final],    "SPIKAN")
plot_helper_kg(1, 1, np.abs(saved_datasets['spikan'][:, :, t_idx_final]    - u_exact_final), "SPIKAN",    is_error=True)
plot_helper_kg(2, 0, saved_datasets['or-spikan'][:, :, t_idx_final], "OR-SPIKAN")
plot_helper_kg(2, 1, np.abs(saved_datasets['or-spikan'][:, :, t_idx_final] - u_exact_final), "OR-SPIKAN", is_error=True)
plot_helper_kg(3, 0, saved_datasets['xor-spikan'][:, :, t_idx_final],"XOR-SPIKAN")
plot_helper_kg(3, 1, np.abs(saved_datasets['xor-spikan'][:, :, t_idx_final]- u_exact_final), "XOR-SPIKAN",is_error=True)


other_modes = ['tanh_or-spikan', 'tanh_xor-spikan', 'sigmoid_or-spikan', 'sigmoid_xor-spikan']
for idx, mode in enumerate(other_modes):
    u_p         = saved_datasets[mode][:, :, t_idx_final]
    title_upper = mode.replace('_', ' ').upper()
    plot_helper_kg(idx, 2, u_p,                        title_upper)
    plot_helper_kg(idx, 3, np.abs(u_p - u_exact_final),title_upper, is_error=True)


final_plot_path = os.path.join(results_dir, 'comparison_kg_4x4_optimized.png')
plt.savefig(final_plot_path,                     dpi=600, bbox_inches='tight')
plt.savefig(final_plot_path.replace('.png', '.pdf'),      bbox_inches='tight')
plt.show()
