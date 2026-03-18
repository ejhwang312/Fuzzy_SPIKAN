import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.interpolate as interp
import glob

timestamp_dir = './data/20260316_131914'
results_dir = './results/20260316_131914'
Re = 400.0
os.makedirs(results_dir, exist_ok=True)

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})



def format_mode_name(mode_str):
    if mode_str == 'spikan':
        return 'SPIKAN'
    elif mode_str == 'or-spikan':
        return 'OR-SPIKAN'
    elif mode_str == 'xor-spikan':
        return 'XOR-SPIKAN'
    elif mode_str == 'tanh_or-spikan':
        return 'OR-SPIKAN with tanh'
    elif mode_str == 'tanh_xor-spikan':
        return 'XOR-SPIKAN with tanh'
    elif mode_str == 'sigmoid_or-spikan':
        return 'OR-SPIKAN with sigmoid'
    elif mode_str == 'sigmoid_xor-spikan':
        return 'XOR-SPIKAN with sigmoid'
    return mode_str

def set_tick_format(ax, cbar=None, decimals=2):
    formatter = ticker.FormatStrFormatter(f'%.{decimals}f')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(formatter)


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
        data, method=interp_func, bounds_error=False, fill_value=np.nan
    )
    XX, YY = np.meshgrid(x_target_unique, y_target_unique)
    points = np.column_stack((YY.ravel(), XX.ravel()))
    interpolated = interpolator(points).reshape(len(y_target_unique), len(x_target_unique))
    return interpolated


def convert_data(data):
    mesh = {k: convert_masked_to_numpy(v) for k, v in data['mesh'].items()}
    field_vars = {k: convert_masked_to_numpy(v) for k, v in data['field_variables'].items()}
    return {'mesh': mesh, 'field_variables': field_vars}



def plot_centerline_comparison(solutions, reference_fvm_data, Re):
    ref_x = np.array(
        [1.00000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047, 0.5000, 0.2344, 0.2266, 0.1563, 0.0938,
         0.0781, 0.0703, 0.0625, 0.0000])
    ref_y = np.array(
        [1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
         0.0625, 0.0547, 0.0000])

    if Re == 100.:
        ref_u = np.array(
            [1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641, -0.20581, -0.21090, -0.15662,
             -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000])
        ref_v = np.array(
            [0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533, 0.05454, 0.17527, 0.17507,
             0.16077, 0.12317, 0.10890, 0.10091, 0.09233, 0.00000])
    elif Re == 400.:
        ref_u = np.array(
            [1.00000, 0.75837, 0.68439, 0.61756, 0.55892, 0.29093, 0.16256, 0.02135, -0.11477, -0.17119, -0.32726,
             -0.24299, -0.14612, -0.10338, -0.09266, -0.08186, 0.00000])
        ref_v = np.array(
            [0.00000, -0.12146, -0.15663, -0.19254, -0.22847, -0.23827, -0.44993, -0.38598, 0.05186, 0.30174, 0.30203,
             0.28124, 0.22965, 0.20920, 0.19713, 0.18360, 0.00000])

    X_fvm = reference_fvm_data['mesh']['x_mesh']
    Y_fvm = reference_fvm_data['mesh']['y_mesh']
    U_fvm = reference_fvm_data['field_variables']['u']
    V_fvm = reference_fvm_data['field_variables']['v']

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    centerline_fvm_y = Y_fvm.shape[0] // 2
    x_fvm_line = X_fvm[centerline_fvm_y, :]
    v_fvm_line = V_fvm[centerline_fvm_y, :]

    centerline_fvm_x = X_fvm.shape[1] // 2
    y_fvm_line = Y_fvm[:, centerline_fvm_x]
    u_fvm_line = U_fvm[:, centerline_fvm_x]

    ax[0].scatter(ref_x, ref_v, color='k', s=30, label='Reference (Ghia et al., 1982)')
    ax[0].plot(x_fvm_line, v_fvm_line, color='purple', linestyle='--', linewidth=2, label='Reference (FVM)')

    ax[1].scatter(ref_u, ref_y, color='k', s=30, label='Reference (Ghia et al., 1982)')
    ax[1].plot(u_fvm_line, y_fvm_line, color='purple', linestyle='--', linewidth=2, label='Reference (FVM)')

    colors = plt.cm.tab10(np.linspace(0, 1, len(solutions)))

    for (name, data), color in zip(solutions.items(), colors):
        formatted_name = format_mode_name(name) # 이름 변환
        X = data['mesh']['x_mesh']
        Y = data['mesh']['y_mesh']
        U = data['field_variables']['u']
        V = data['field_variables']['v']

        c_y = Y.shape[0] // 2
        ax[0].plot(X[c_y, :], V[c_y, :], color=color, linewidth=2, label=formatted_name)

        c_x = X.shape[1] // 2
        ax[1].plot(U[:, c_x], Y[:, c_x], color=color, linewidth=2, label=formatted_name)

    ax[0].set_xlabel(r'$x/L$')
    ax[0].set_ylabel(r'$v/U_0$')
    ax[0].set_title(r'$v$ on Horizontal Centerline')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(frameon=True, fancybox=True, loc='best', fontsize=10)

    ax[1].set_xlabel(r'$u/U_0$')
    ax[1].set_ylabel(r'$y/L$')
    ax[1].set_title(r'$u$ on Vertical Centerline')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(frameon=True, fancybox=True, loc='best', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'centerline_comparison_Re{Re}.png'), dpi=600, bbox_inches='tight')
    plt.show()



def plot_field_comparison(solutions, reference_fvm_data, Re):
    X_fvm = reference_fvm_data['mesh']['x_mesh']
    Y_fvm = reference_fvm_data['mesh']['y_mesh']
    U_fvm = reference_fvm_data['field_variables']['u']
    V_fvm = reference_fvm_data['field_variables']['v']
    p_fvm = reference_fvm_data['field_variables']['p']
    p_fvm_plot = (p_fvm - np.mean(p_fvm)) / np.max(np.abs(p_fvm - np.mean(p_fvm)))

    n_rows = len(solutions) + 1
    fig, axs = plt.subplots(n_rows, 4, figsize=(24, 6 * n_rows))

    # Row 0: FVM
    im00 = axs[0, 0].contourf(X_fvm, Y_fvm, U_fvm, levels=50, cmap='RdBu_r')
    cbar00 = plt.colorbar(im00, ax=axs[0, 0], fraction=0.046, pad=0.04)
    cbar00.set_label(r'$u$')
    axs[0, 0].set_title(r'Reference $u$, FVM')

    im01 = axs[0, 1].contourf(X_fvm, Y_fvm, V_fvm, levels=50, cmap='RdBu_r')
    cbar01 = plt.colorbar(im01, ax=axs[0, 1], fraction=0.046, pad=0.04)
    cbar01.set_label(r'$v$')
    axs[0, 1].set_title(r'Reference $v$, FVM')

    im02 = axs[0, 2].contourf(X_fvm, Y_fvm, p_fvm_plot, levels=50, cmap='RdBu_r')
    cbar02 = plt.colorbar(im02, ax=axs[0, 2], fraction=0.046, pad=0.04)
    cbar02.set_label(r'$p$')
    axs[0, 2].set_title(r'Reference $p$, FVM')

    axs[0, 3].streamplot(X_fvm, Y_fvm, U_fvm, V_fvm, density=2, color='black')
    axs[0, 3].set_title(r'Streamlines, Reference (FVM)')

    for i in range(4):
        axs[0, i].set_xlabel(r'$x/L$')
        axs[0, i].set_ylabel(r'$y/L$')
        axs[0, i].set_aspect('equal')
        axs[0, i].set_xlim(0, 1)
        axs[0, i].set_ylim(0, 1)
        if i < 3: set_tick_format(axs[0, i], [cbar00, cbar01, cbar02][i])

    # Row 1~N: Models
    for idx, (name, data) in enumerate(solutions.items(), start=1):
        formatted_name = format_mode_name(name) # 이름 변환
        X = data['mesh']['x_mesh']
        Y = data['mesh']['y_mesh']
        U = data['field_variables']['u']
        V = data['field_variables']['v']
        p = data['field_variables']['p']
        p_plot = (p - np.mean(p)) / np.max(np.abs(p - np.mean(p)))

        im0 = axs[idx, 0].contourf(X, Y, U, levels=50, cmap='RdBu_r')
        cbar0 = plt.colorbar(im0, ax=axs[idx, 0], fraction=0.046, pad=0.04)
        cbar0.set_label(r'$u$')
        axs[idx, 0].set_title(f'Predicted $u$, {formatted_name}')

        im1 = axs[idx, 1].contourf(X, Y, V, levels=50, cmap='RdBu_r')
        cbar1 = plt.colorbar(im1, ax=axs[idx, 1], fraction=0.046, pad=0.04)
        cbar1.set_label(r'$v$')
        axs[idx, 1].set_title(f'Predicted $v$, {formatted_name}')

        im2 = axs[idx, 2].contourf(X, Y, p_plot, levels=50, cmap='RdBu_r')
        cbar2 = plt.colorbar(im2, ax=axs[idx, 2], fraction=0.046, pad=0.04)
        cbar2.set_label(r'$p$')
        axs[idx, 2].set_title(f'Predicted $p$, {formatted_name}')

        U_stream = np.nan_to_num(U, nan=0.0)
        V_stream = np.nan_to_num(V, nan=0.0)
        axs[idx, 3].streamplot(X, Y, U_stream, V_stream, density=2, color='black')
        axs[idx, 3].set_title(f'Streamlines, {formatted_name}')

        for j in range(4):
            axs[idx, j].set_xlabel(r'$x/L$')
            axs[idx, j].set_ylabel(r'$y/L$')
            axs[idx, j].set_aspect('equal')
            axs[idx, j].set_xlim(0, 1)
            axs[idx, j].set_ylim(0, 1)
            if j < 3: set_tick_format(axs[idx, j], [cbar0, cbar1, cbar2][j])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'field_comparison_Re{Re}.png'), dpi=600, bbox_inches='tight')
    plt.show()



def plot_absolute_difference(solutions, reference_fvm_data, Re):
    X_fvm = reference_fvm_data['mesh']['x_mesh']
    Y_fvm = reference_fvm_data['mesh']['y_mesh']
    U_fvm = reference_fvm_data['field_variables']['u']
    V_fvm = reference_fvm_data['field_variables']['v']
    p_fvm = reference_fvm_data['field_variables']['p']

    grid_results = create_regular_grid(X_fvm, Y_fvm)
    X_fvm_reg = grid_results[0]
    Y_fvm_reg = grid_results[1]

    n_rows = len(solutions)
    fig, axs = plt.subplots(n_rows, 3, figsize=(18, 6 * n_rows))

    if n_rows == 1: axs = np.expand_dims(axs, axis=0)

    for idx, (name, data) in enumerate(solutions.items()):
        formatted_name = format_mode_name(name) # 이름 변환
        X = data['mesh']['x_mesh']
        Y = data['mesh']['y_mesh']
        U = data['field_variables']['u']
        V = data['field_variables']['v']
        p = data['field_variables']['p']

        U_interp = interpolate_to_fvm_grid(U, (X, Y), (X_fvm, Y_fvm))
        V_interp = interpolate_to_fvm_grid(V, (X, Y), (X_fvm, Y_fvm))
        p_interp = interpolate_to_fvm_grid(p, (X, Y), (X_fvm, Y_fvm))

        err_u = np.abs(U_interp - U_fvm)
        err_v = np.abs(V_interp - V_fvm)

        p_fvm_norm = (p_fvm - np.mean(p_fvm)) / np.max(np.abs(p_fvm - np.mean(p_fvm)))
        p_norm = (p_interp - np.mean(p_interp)) / np.max(np.abs(p_interp - np.mean(p_interp)))
        err_p = np.abs(p_norm - p_fvm_norm)

        errors = [(err_u, 'u'), (err_v, 'v'), (err_p, 'p')]

        for j, (err, var) in enumerate(errors):
            im = axs[idx, j].contourf(X_fvm_reg, Y_fvm_reg, err, levels=50, cmap='RdBu_r')
            cbar = plt.colorbar(im, ax=axs[idx, j], fraction=0.046, pad=0.04)
            cbar.set_label(rf'$|{var}_{{{formatted_name}}} - {var}_{{FVM}}|$')
            axs[idx, j].set_title(f'Absolute Difference in ${var}$, {formatted_name} vs FVM')
            axs[idx, j].set_xlabel(r'$x/L$')
            axs[idx, j].set_ylabel(r'$y/L$')
            axs[idx, j].set_aspect('equal')
            axs[idx, j].set_xlim(0, 1)
            axs[idx, j].set_ylim(0, 1)
            set_tick_format(axs[idx, j], cbar)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'absolute_difference_comparison_Re{Re}.png'), dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    ref_fvm_path = f'./data/reference_data/2d_ns_reference_Re_{int(Re)}_nx256_ny256.npy'
    if not os.path.exists(ref_fvm_path):
        raise FileNotFoundError(f"FVM Reference data not found at: {ref_fvm_path}")

    reference_fvm_data = convert_data(np.load(ref_fvm_path, allow_pickle=True).item())

    modes_to_test = [
        'spikan', 'or-spikan', 'xor-spikan',
        'tanh_or-spikan', 'tanh_xor-spikan',
        'sigmoid_or-spikan', 'sigmoid_xor-spikan'
    ]

    solutions = {}
    for mode in modes_to_test:
        search_pattern = os.path.join(timestamp_dir, f'2d_ns_{mode}_Re_{Re}_*.npy')
        file_list = glob.glob(search_pattern)
        if len(file_list) > 0:
            file_path = file_list[0]
            solutions[mode] = convert_data(np.load(file_path, allow_pickle=True).item())
            print(f"[{mode}] Loaded: {file_path}")


    if len(solutions) > 0:
        plot_centerline_comparison(solutions, reference_fvm_data, Re)

        plot_field_comparison(solutions, reference_fvm_data, Re)

        plot_absolute_difference(solutions, reference_fvm_data, Re)
