import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import glob

timestamp_dir = './data/20260315_164028'
results_dir = './results/20260315_164028'
os.makedirs(results_dir, exist_ok=True)


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


modes_to_test = [
    'spikan', 'or-spikan', 'xor-spikan',
    'tanh_or-spikan', 'tanh_xor-spikan',
    'sigmoid_or-spikan', 'sigmoid_xor-spikan'
]

saved_datasets = {}
X_mesh, Y_mesh = None, None
a1, a2 = 1.0, 4.0

for mode in modes_to_test:
    search_pattern = os.path.join(timestamp_dir, f'2d_helmholtz_{mode}_*.npy')
    file_list = glob.glob(search_pattern)
    if len(file_list) > 0:
        file_path = file_list[0]
        data = np.load(file_path, allow_pickle=True).item()


        saved_datasets[mode] = {'u': data['field_variables']['u']}


        if X_mesh is None:
            X_mesh = data['mesh']['x_mesh']
            Y_mesh = data['mesh']['y_mesh']
            a1 = data['parameters']['a1']
            a2 = data['parameters']['a2']

        print(f"[{mode}] Loaded: {file_path}")



if len(saved_datasets) > 0:

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
    gs = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.45,
                           left=0.08, right=0.92, top=0.95, bottom=0.05)


    def plot_helper(row, col, u_val, title_base, is_error=False):
        ax = plt.subplot(gs[row, col])
        cmap = 'RdBu_r'
        if is_error:
            title = f'Absolute Error,\n{title_base}'
            label = r'$|u_{\text{pred}} - u_{\text{exact}}|$'
            im = ax.contourf(X_mesh, Y_mesh, u_val, levels=50, cmap=cmap)
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
    if 'spikan' in saved_datasets:
        plot_helper(1, 0, saved_datasets['spikan']['u'], "SPIKAN")
        plot_helper(1, 1, np.abs(saved_datasets['spikan']['u'] - u_exact_vis), "SPIKAN", is_error=True)
    if 'or-spikan' in saved_datasets:
        plot_helper(2, 0, saved_datasets['or-spikan']['u'], "OR-SPIKAN")
        plot_helper(2, 1, np.abs(saved_datasets['or-spikan']['u'] - u_exact_vis), "OR-SPIKAN", is_error=True)
    if 'xor-spikan' in saved_datasets:
        plot_helper(3, 0, saved_datasets['xor-spikan']['u'], "XOR-SPIKAN")
        plot_helper(3, 1, np.abs(saved_datasets['xor-spikan']['u'] - u_exact_vis), "XOR-SPIKAN", is_error=True)

    other_modes = ['tanh_or-spikan', 'tanh_xor-spikan', 'sigmoid_or-spikan', 'sigmoid_xor-spikan']
    for idx, mode in enumerate(other_modes):
        if mode in saved_datasets:
            u_p = saved_datasets[mode]['u']
            title_upper = format_mode_name(mode)
            plot_helper(idx, 2, u_p, title_upper)
            plot_helper(idx, 3, np.abs(u_p - u_exact_vis), title_upper, is_error=True)

    final_plot_path = os.path.join(results_dir, 'comparison_helmholtz2d_4x4_optimized_dpi_150.png')
    plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(final_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()
