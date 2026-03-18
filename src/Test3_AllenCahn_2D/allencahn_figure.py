import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from scipy.interpolate import RegularGridInterpolator
import glob


timestamp_dir = './data/20260312_105323'
results_dir = './results/20260312_105323'
os.makedirs(results_dir, exist_ok=True)


def set_tick_format(ax, cbar=None, decimals=2):
    fmt = ticker.FormatStrFormatter(f'%.{decimals}f')
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    if cbar is not None:
        cbar.ax.yaxis.set_major_formatter(fmt)


def interpolate_to_reference_grid(data, ref_t, ref_x):
    interp_func = RegularGridInterpolator(
        (np.array(data['t']), np.array(data['x'])),
        np.array(data['u']), method='linear', bounds_error=False, fill_value=None
    )
    T_ref, X_ref = np.meshgrid(ref_t, ref_x, indexing='ij')
    points = np.stack((T_ref.flatten(), X_ref.flatten()), axis=1)
    return interp_func(points).reshape(T_ref.shape)



ref_save_path = os.path.join(timestamp_dir, 'allen_cahn_ref.npy')
spectral_data = np.load(ref_save_path, allow_pickle=True).item()
T_spectral, X_spectral = np.meshgrid(spectral_data['t'], spectral_data['x'], indexing='ij')
u_exact = spectral_data['u']

modes_to_test = [
    'spikan', 'or-spikan', 'xor-spikan',
    'tanh_or-spikan', 'tanh_xor-spikan',
    'sigmoid_or-spikan', 'sigmoid_xor-spikan'
]

interpolated_solutions = {}
errors = {}

for mode in modes_to_test:
    search_pattern = os.path.join(timestamp_dir, f'allen_cahn_{mode}_*.npy')
    file_list = glob.glob(search_pattern)



    file_path = file_list[0]
    data = np.load(file_path, allow_pickle=True).item()

    u_interp = interpolate_to_reference_grid(data, spectral_data['t'], spectral_data['x'])
    abs_err = np.abs(u_interp - u_exact)

    interpolated_solutions[mode] = u_interp
    errors[mode] = abs_err

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 12,
})

fig = plt.figure(figsize=(21, 18))
gs = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.45,
                       left=0.08, right=0.92, top=0.95, bottom=0.05)


def plot_helper_ac(row, col, u_val, title_base, is_error=False):
    ax = plt.subplot(gs[row, col])
    cmap = 'RdBu_r'
    if is_error:
        title = f'Absolute Error,\n{title_base}'
        label = r'$|u_{\text{pred}} - u_{\text{exact}}|$'
        im = ax.contourf(T_spectral, X_spectral, u_val, levels=50, cmap=cmap)
    else:
        title = "Reference Solution" if title_base == "REF" else f'Predicted $u$,\n{title_base}'
        label = r'$u$'
        im = ax.contourf(T_spectral, X_spectral, u_val, levels=50, cmap=cmap, vmin=-1, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    ax.set_title(title)
    ax.set_aspect('auto')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$x$')
    set_tick_format(ax, cbar, decimals=2)


plot_helper_ac(0, 0, u_exact, "REF")

if 'spikan' in interpolated_solutions:
    plot_helper_ac(1, 0, interpolated_solutions['spikan'], "SPIKAN")
    plot_helper_ac(1, 1, errors['spikan'], "SPIKAN", is_error=True)

if 'or-spikan' in interpolated_solutions:
    plot_helper_ac(2, 0, interpolated_solutions['or-spikan'], "OR-SPIKAN")
    plot_helper_ac(2, 1, errors['or-spikan'], "OR-SPIKAN", is_error=True)

if 'xor-spikan' in interpolated_solutions:
    plot_helper_ac(3, 0, interpolated_solutions['xor-spikan'], "XOR-SPIKAN")
    plot_helper_ac(3, 1, errors['xor-spikan'], "XOR-SPIKAN", is_error=True)

other_modes = ['tanh_or-spikan', 'tanh_xor-spikan', 'sigmoid_or-spikan', 'sigmoid_xor-spikan']
for idx, mode in enumerate(other_modes):
    if mode in interpolated_solutions:
        title_upper = mode.replace('_', ' ').upper()
        plot_helper_ac(idx, 2, interpolated_solutions[mode], title_upper)
        plot_helper_ac(idx, 3, errors[mode], title_upper, is_error=True)


final_plot_path = os.path.join(results_dir, 'comparison_allencahn_4x4_optimized_dpi_150.png')
plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
plt.savefig(final_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
plt.show()