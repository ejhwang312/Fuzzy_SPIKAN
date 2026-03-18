import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import jax.numpy as jnp
import jax
import time
from datetime import datetime
from jax import vmap, jit, jacfwd
import optax
from functools import partial
import numpy as np
from tqdm import trange
from jax import random
import matplotlib.pyplot as plt
import sys
import scipy.interpolate as interp
sys.path.insert(0, '../')

from KAN import KAN


def convert_masked_to_numpy(arr):
    if isinstance(arr, np.ma.MaskedArray):
        return arr.filled(np.nan)
    return np.array(arr)


def create_regular_grid(x, y):
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    X, Y = np.meshgrid(x_unique, y_unique)
    return X, Y, x_unique, y_unique


def interpolate_to_fvm_grid(data, source_mesh, target_mesh, interp_func='linear'):
    x_source, y_source = source_mesh
    x_target, y_target = target_mesh

    _, _, x_source_unique, y_source_unique = create_regular_grid(x_source, y_source)
    _, _, x_target_unique, y_target_unique = create_regular_grid(x_target, y_target)

    if data.shape != (len(y_source_unique), len(x_source_unique)):
        data = data.reshape(len(y_source_unique), len(x_source_unique))

    interpolator = interp.RegularGridInterpolator(
        (y_source_unique, x_source_unique),
        data,
        method=interp_func,
        bounds_error=False,
        fill_value=np.nan
    )

    XX, YY = np.meshgrid(x_target_unique, y_target_unique)
    points = np.column_stack((YY.ravel(), XX.ravel()))

    interpolated = interpolator(points).reshape(len(y_target_unique), len(x_target_unique))
    return interpolated


def calculate_l2_difference(pred, ref, is_pressure=False):
    pred_flat = pred.flatten()
    ref_flat = ref.flatten()
    if is_pressure:
        pred_flat = pred_flat - np.mean(pred_flat)
        ref_flat = ref_flat - np.mean(ref_flat)
    l2_diff = np.sqrt(np.sum((pred_flat - ref_flat) ** 2))
    l2_ref = np.sqrt(np.sum(ref_flat ** 2))
    return 100 * (l2_diff / l2_ref) if l2_ref != 0 else float('inf')



def create_interior_points(L_range, H_range, nx, ny):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)
    return x, y


def create_boundary_points(L_range, H_range, nx, ny):
    x = np.linspace(L_range[0], L_range[1], nx)
    y = np.linspace(H_range[0], H_range[1], ny)

    x_left = jnp.array([L_range[0]])
    y_left = y

    x_top = x
    y_top = jnp.array([H_range[1]])

    x_right = jnp.array([L_range[1]])
    y_right = y

    x_bottom = x
    y_bottom = jnp.array([H_range[0]])

    return (x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom)


# Define the domain
L_range = (0.0, 1.0)
H_range = (0.0, 1.0)  # Square cavity

# Create interior points using nx and ny
nx, ny = 200, 200 # Grid resolution
x_interior, y_interior = create_interior_points(L_range, H_range, nx, ny)

# Create boundary points using nx and ny
(x_left, y_left), (x_top, y_top), (x_right, y_right), (x_bottom, y_bottom) = create_boundary_points(L_range, H_range,
                                                                                                    nx, ny)


def plot_domain_setup(x_interior, y_interior, x_left, y_left,
                      x_top, y_top, x_right, y_right,
                      x_bottom, y_bottom):
    plt.figure(figsize=(6, 6))

    X, Y = np.meshgrid(x_interior, y_interior)
    plt.scatter(X, Y, s=1, alpha=0.5, label='Interior')

    plt.scatter(np.full_like(y_left, x_left[0]), y_left, s=5, c='r', alpha=0.5, label='Left')
    plt.scatter(x_top, np.full_like(x_top, y_top[0]), s=5, c='g', alpha=0.5, label='Top (Lid)')
    plt.scatter(np.full_like(y_right, x_right[0]), y_right, s=5, c='b', alpha=0.5, label='Right')
    plt.scatter(x_bottom, np.full_like(x_bottom, y_bottom[0]), s=5, c='m', alpha=0.5, label='Bottom')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


class SF_KAN_Separable:
    def __init__(self, layer_dims, init_lr, Re=100, k=5, r=1, mode='spikan'):
        self.input_size = layer_dims[0]
        self.out_size = layer_dims[-1]
        self.r = r
        self.layer_dims = [self.input_size] + layer_dims[1:-1] + [self.r * self.out_size]

        self.model_x = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, add_bias=True,
                           grid_e=0.02, j='0', mode=mode)
        self.model_y = KAN(layer_dims=self.layer_dims, k=k, const_spl=False, const_res=False, add_bias=True,
                           grid_e=0.02, j='0', mode=mode)

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
        old_shape = mu_old.shape
        size = old_shape[0]
        old_j = old_shape[1]
        new_j = new_shape[1]

        old_indices = jnp.linspace(0, old_j - 1, old_j)
        new_indices = jnp.linspace(0, old_j - 1, new_j)

        interpolate_fn = lambda old_row: jnp.interp(new_indices, old_indices, old_row)

        mu_new = vmap(interpolate_fn)(mu_old)
        nu_new = vmap(interpolate_fn)(nu_old)

        return mu_new, nu_new

    def smooth_state_transition(self, old_state, params):
        adam_count = old_state[0].count
        adam_mu, adam_nu = old_state[0].mu, old_state[0].nu

        layer_keys = {k for k in adam_mu.keys() if k.startswith('layers_')}

        for key in layer_keys:
            c_shape = params[key]['c_basis'].shape
            mu_new0, nu_new0 = self.interpolate_moments(adam_mu[key]['c_basis'], adam_nu[key]['c_basis'], c_shape)
            adam_mu[key]['c_basis'], adam_nu[key]['c_basis'] = mu_new0, nu_new0

        adam_state = optax.ScaleByAdamState(adam_count, adam_mu, adam_nu)
        extra_state = optax.ScaleByScheduleState(adam_count)
        new_state = (adam_state, extra_state)

        return new_state

    def predict(self, x, y):
        variables_x, variables_y = self.variables_x, self.variables_y
        preds, _ = self.forward_pass(variables_x, variables_y, x, y)
        return preds

    @partial(jit, static_argnums=(0,))
    def forward_pass(self, variables_x, variables_y, x, y):
        preds_x, spl_regs_x = self.model_x.apply(variables_x, x[:, None])
        preds_y, spl_regs_y = self.model_y.apply(variables_y, y[:, None])

        preds_x = preds_x.reshape(-1, self.out_size, self.r)
        preds_y = preds_y.reshape(-1, self.out_size, self.r)
        preds = jnp.einsum('ijk,ljk->ilj', preds_x, preds_y)

        spl_regs = spl_regs_x + spl_regs_y

        return preds, spl_regs

    @partial(jit, static_argnums=(0,))
    def loss(self, params_x, params_y, state_x, state_y, *args):
        variables_x = {'params': params_x, 'state': state_x}
        variables_y = {'params': params_y, 'state': state_y}
        return self.loss_fn(variables_x, variables_y, *args)

    @partial(jit, static_argnums=(0,))
    def train_step(self, params_x, params_y, state_x, state_y, opt_state_x, opt_state_y, *args):
        (loss_value, (physics_loss, boundary_loss)), grads = jax.value_and_grad(self.loss, has_aux=True,
                                                                                argnums=(0, 1))(
            params_x, params_y, state_x, state_y, *args
        )
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
                    'Total Loss': f"{loss_value:.4e}",
                    'Physics Loss': f"{physics_loss:.4e}",
                    'Boundary Loss': f"{boundary_loss:.4e}"
                })

        self.variables_x = {'params': params_x, 'state': state_x}
        self.variables_y = {'params': params_y, 'state': state_y}
        return loss_history


class Cavity_SF_KAN_Separable(SF_KAN_Separable):
    def __init__(self, *args, Re=100.0, r=10, mode='spikan', **kwargs):
        super().__init__(*args, r=r, mode=mode, **kwargs)
        self.Re = Re

    @partial(jit, static_argnums=(0,))
    def loss_fn(self, variables_x, variables_y, x_interior, y_interior, x_left, y_left, x_top, y_top, x_right, y_right,
                x_bottom, y_bottom):
        residuals = self.compute_residuals(variables_x, variables_y, x_interior, y_interior)
        physics_loss = jnp.mean(jnp.square(residuals))

        preds_left, _ = self.forward_pass(variables_x, variables_y, x_left, y_left)
        preds_top, _ = self.forward_pass(variables_x, variables_y, x_top, y_top)
        preds_right, _ = self.forward_pass(variables_x, variables_y, x_right, y_right)
        preds_bottom, _ = self.forward_pass(variables_x, variables_y, x_bottom, y_bottom)

        left_loss = jnp.mean(jnp.square(preds_left[..., 0]) + jnp.square(preds_left[..., 1]))
        top_loss = jnp.mean(jnp.square(preds_top[..., 0] - 1.0) + jnp.square(preds_top[..., 1]))
        right_loss = jnp.mean(jnp.square(preds_right[..., 0]) + jnp.square(preds_right[..., 1]))
        bottom_loss = jnp.mean(jnp.square(preds_bottom[..., 0]) + jnp.square(preds_bottom[..., 1]))

        boundary_loss = left_loss + top_loss + right_loss + bottom_loss
        total_loss = physics_loss + boundary_loss

        return total_loss, (physics_loss, boundary_loss)

    @partial(jit, static_argnums=(0,))
    def compute_residuals(self, variables_x, variables_y, x_interior, y_interior):
        def model_x_func(x):
            x_feat = self.model_x.apply(variables_x, x.reshape(-1, 1))[0]
            x_feat = x_feat.reshape(self.out_size, self.r)
            return x_feat

        def model_y_func(y):
            y_feat = self.model_y.apply(variables_y, y.reshape(-1, 1))[0]
            y_feat = y_feat.reshape(self.out_size, self.r)
            return y_feat

        def model_x_grad(x):
            return jacfwd(model_x_func)(x)

        def model_y_grad(y):
            return jacfwd(model_y_func)(y)

        def model_x_hess(x):
            return jacfwd(jacfwd(model_x_func))(x)

        def model_y_hess(y):
            return jacfwd(jacfwd(model_y_func))(y)

        x_feats = vmap(model_x_func)(x_interior)
        y_feats = vmap(model_y_func)(y_interior)
        x_grads = vmap(model_x_grad)(x_interior)
        y_grads = vmap(model_y_grad)(y_interior)
        x_hess = vmap(model_x_hess)(x_interior)
        y_hess = vmap(model_y_hess)(y_interior)

        u_x, v_x, p_x = x_feats[:, 0, :], x_feats[:, 1, :], x_feats[:, 2, :]
        u_y, v_y, p_y = y_feats[:, 0, :], y_feats[:, 1, :], y_feats[:, 2, :]

        du_x_dx, dv_x_dx, dp_x_dx = x_grads[:, 0, :], x_grads[:, 1, :], x_grads[:, 2, :]
        du_y_dy, dv_y_dy, dp_y_dy = y_grads[:, 0, :], y_grads[:, 1, :], y_grads[:, 2, :]

        d2u_x_dx2, d2v_x_dx2 = x_hess[:, 0, :], x_hess[:, 1, :]
        d2u_y_dy2, d2v_y_dy2 = y_hess[:, 0, :], y_hess[:, 1, :]

        u = jnp.einsum('ir,jr->ij', u_x, u_y)
        v = jnp.einsum('ir,jr->ij', v_x, v_y)
        p = jnp.einsum('ir,jr->ij', p_x, p_y)

        du_dx = jnp.einsum('ir,jr->ij', du_x_dx, u_y)
        du_dy = jnp.einsum('ir,jr->ij', u_x, du_y_dy)
        dv_dx = jnp.einsum('ir,jr->ij', dv_x_dx, v_y)
        dv_dy = jnp.einsum('ir,jr->ij', v_x, dv_y_dy)
        dp_dx = jnp.einsum('ir,jr->ij', dp_x_dx, p_y)
        dp_dy = jnp.einsum('ir,jr->ij', p_x, dp_y_dy)

        d2u_dx2 = jnp.einsum('ir,jr->ij', d2u_x_dx2, u_y)
        d2u_dy2 = jnp.einsum('ir,jr->ij', u_x, d2u_y_dy2)
        d2v_dx2 = jnp.einsum('ir,jr->ij', d2v_x_dx2, v_y)
        d2v_dy2 = jnp.einsum('ir,jr->ij', v_x, d2v_y_dy2)

        continuity = du_dx + dv_dy
        momentum_x = u * du_dx + v * du_dy + dp_dx - (1 / self.Re) * (d2u_dx2 + d2u_dy2)
        momentum_y = u * dv_dx + v * dv_dy + dp_dy - (1 / self.Re) * (d2v_dx2 + d2v_dy2)

        return jnp.stack([continuity, momentum_x, momentum_y], axis=-1)


layer_dims = [1, 5, 5, 3]  # Input dim is always 1, output dim is 3 (u, v, p)
init_lr = 1e-3
Re = 400.0
k = 5
r = 5
num_epochs = 100000

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
save_dir = os.path.join('./data', timestamp)
results_dir = os.path.join('./results', timestamp)
os.makedirs(save_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
print(f"저장 폴더: {save_dir}")

modes_to_test = [
    'spikan',
    'or-spikan',
    'xor-spikan',
    'tanh_or-spikan',
    'tanh_xor-spikan',
    'sigmoid_or-spikan',
    'sigmoid_xor-spikan',
]

times_per_iter = {}
l2_errors = {}
max_errors = {}
mean_errors = {}

ref_fvm_path = f'./data/reference_data/2d_ns_reference_Re_{int(Re)}_nx256_ny256.npy'
if os.path.exists(ref_fvm_path):
    reference_fvm_data = np.load(ref_fvm_path, allow_pickle=True).item()
    X_fvm = convert_masked_to_numpy(reference_fvm_data['mesh']['x_mesh'])
    Y_fvm = convert_masked_to_numpy(reference_fvm_data['mesh']['y_mesh'])
    U_fvm = convert_masked_to_numpy(reference_fvm_data['field_variables']['u'])
    V_fvm = convert_masked_to_numpy(reference_fvm_data['field_variables']['v'])
    p_fvm = convert_masked_to_numpy(reference_fvm_data['field_variables']['p'])
    has_ref = True
else:
    has_ref = False

for current_mode in modes_to_test:
    model = Cavity_SF_KAN_Separable(
        layer_dims=layer_dims,
        init_lr=init_lr,
        Re=Re,
        k=k,
        r=r,
        mode=current_mode,
    )

    start_time = time.time()

    loss_history = model.train(
        num_epochs,
        x_interior, y_interior,
        x_left, y_left,
        x_top, y_top,
        x_right, y_right,
        x_bottom, y_bottom
    )

    end_time = time.time()
    ms_per_iter = (end_time - start_time) / num_epochs * 1000
    times_per_iter[current_mode] = ms_per_iter

    nx_plot, ny_plot = 100, 100
    x_plot = jnp.linspace(L_range[0], L_range[1], nx_plot)
    y_plot = jnp.linspace(L_range[0], L_range[1], ny_plot)
    X_mesh, Y_mesh = jnp.meshgrid(x_plot, y_plot)

    uvp_pred = model.predict(x_plot, y_plot)
    U = np.array(uvp_pred[:, :, 0]).T
    V = np.array(uvp_pred[:, :, 1]).T
    P = np.array(uvp_pred[:, :, 2]).T
    vmag = np.sqrt(U ** 2 + V ** 2)

    if has_ref:
        source_mesh = (X_mesh, Y_mesh)
        target_mesh = (X_fvm, Y_fvm)

        U_interp = interpolate_to_fvm_grid(U, source_mesh, target_mesh)
        V_interp = interpolate_to_fvm_grid(V, source_mesh, target_mesh)
        p_interp = interpolate_to_fvm_grid(P, source_mesh, target_mesh)

        # L2 Error (u, v, p)
        l2_errors[current_mode] = {
            'u': calculate_l2_difference(U_interp, U_fvm),
            'v': calculate_l2_difference(V_interp, V_fvm),
            'p': calculate_l2_difference(p_interp, p_fvm, is_pressure=True)
        }

        # Absolute Error (u, v, p)
        err_u = np.abs(U_interp - U_fvm)
        err_v = np.abs(V_interp - V_fvm)

        p_fvm_norm = (p_fvm - np.mean(p_fvm)) / np.max(np.abs(p_fvm - np.mean(p_fvm)))
        p_p_norm = (p_interp - np.mean(p_interp)) / np.max(np.abs(p_interp - np.mean(p_interp)))
        err_p = np.abs(p_p_norm - p_fvm_norm)

        max_errors[current_mode] = {'u': np.max(err_u), 'v': np.max(err_v), 'p': np.max(err_p)}
        mean_errors[current_mode] = {'u': np.mean(err_u), 'v': np.mean(err_v), 'p': np.mean(err_p)}
    else:
        l2_errors[current_mode] = {'u': 0.0, 'v': 0.0, 'p': 0.0}
        max_errors[current_mode] = {'u': 0.0, 'v': 0.0, 'p': 0.0}
        mean_errors[current_mode] = {'u': 0.0, 'v': 0.0, 'p': 0.0}

    output_data = {
        'mesh': {'x_mesh': X_mesh, 'y_mesh': Y_mesh, 'L_range': L_range},
        'field_variables': {'u': U, 'v': V, 'vmag': vmag, 'p': P},
        'parameters': {'Re': Re},
        'training': {'loss_history': loss_history},
        'ms_per_iter': ms_per_iter
    }

    save_path = os.path.join(
        save_dir,
        f'2d_ns_{current_mode}_Re_{Re}_nx{nx}_ny{ny}_epochs{num_epochs}_{layer_dims}.npy'
    )
    np.save(save_path, output_data)


print("\n─── Evaluation Metrics ───────────────────")
for mode in modes_to_test:
    print(f"[{mode}] Time: {times_per_iter[mode]:.3f} ms/iter")
    print(
        f"   - u: L2 {l2_errors[mode]['u']:.3f}%, Max AE {max_errors[mode]['u']:.4f}, Mean AE {mean_errors[mode]['u']:.4f}")
    print(
        f"   - v: L2 {l2_errors[mode]['v']:.3f}%, Max AE {max_errors[mode]['v']:.4f}, Mean AE {mean_errors[mode]['v']:.4f}")
    print(
        f"   - p: L2 {l2_errors[mode]['p']:.3f}%, Max AE {max_errors[mode]['p']:.4f}, Mean AE {mean_errors[mode]['p']:.4f}")

txt_save_path = os.path.join(results_dir, 'evaluation_summary.txt')
with open(txt_save_path, 'w', encoding='utf-8') as f:
    f.write("=== Hyperparameters ===\n")
    f.write(f"nx (Interior points): {nx}\n")
    f.write(f"ny (Interior points): {ny}\n")
    f.write(f"nx_plot (Plot points): {nx_plot}\n")
    f.write(f"ny_plot (Plot points): {ny_plot}\n")
    f.write(f"Re (Reynolds Number): {Re}\n")
    f.write(f"Layer dims: {layer_dims}\n")
    f.write(f"Learning rate (init_lr): {init_lr}\n")
    f.write(f"k (Spline order): {k}\n")
    f.write(f"r (Rank): {r}\n")
    f.write(f"Num epochs: {num_epochs}\n\n")


    for mode in modes_to_test:
        f.write(f"Mode: {mode}\n")
        f.write(f"  - Time per iter: {times_per_iter[mode]:.3f} ms/iter\n")
        for var in ['u', 'v', 'p']:
            f.write(f"  [{var}]\n")
            f.write(f"    - L2 Error:       {l2_errors[mode][var]:.3f} %\n")
            f.write(f"    - Max Abs Error:  {max_errors[mode][var]:.4f}\n")
            f.write(f"    - Mean Abs Error: {mean_errors[mode][var]:.4f}\n")
        f.write("-" * 40 + "\n")



def count_trainable_params(variables_x, variables_y):
    flat_params_x, _ = jax.tree_util.tree_flatten(variables_x['params'])
    flat_params_y, _ = jax.tree_util.tree_flatten(variables_y['params'])
    total_params_x = sum(p.size for p in flat_params_x)
    total_params_y = sum(p.size for p in flat_params_y)
    return total_params_x + total_params_y