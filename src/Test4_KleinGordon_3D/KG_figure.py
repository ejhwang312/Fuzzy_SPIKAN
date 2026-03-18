import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import glob


timestamp_dir = './data/20260312_135655'
results_dir = './results/20260312_135655'
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
u_exact_all = None
X_ij, Y_ij = None, None
nx, ny, nt = 100, 100, 100
T_range = (0.0, 10.0)

for mode in modes_to_test:
    search_pattern = os.path.join(timestamp_dir, f'2d_kg_{mode}_*.npy')
    file_list = glob.glob(search_pattern)
    if len(file_list) > 0:
        file_path = file_list[0]
        data = np.load(file_path, allow_pickle=True).item()

        saved_datasets[mode] = data['field_variables']['u_pred']

        if u_exact_all is None:
            u_exact_all = data['field_variables']['u_exact']
            nx = data['mesh']['nx']
            ny = data['mesh']['ny']
            nt = data['mesh']['nt']
            L_range = data['mesh']['L_range']
            H_range = data['mesh']['H_range']
            T_range = data['mesh']['T_range']

            x_interior = np.linspace(L_range[0], L_range[1], nx)
            y_interior = np.linspace(H_range[0], H_range[1], ny)
            X_ij, Y_ij = np.meshgrid(x_interior, y_interior, indexing='ij')


if len(saved_datasets) > 0:
    plt.rcParams.update({
        'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 12,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 12,
    })

    t_idx_final = nt - 1
    t_final_val = T_range[1]
    u_exact_final = u_exact_all[:, :, t_idx_final]

    fig = plt.figure(figsize=(21, 18))
    gs = gridspec.GridSpec(4, 4, wspace=0.35, hspace=0.45,
                           left=0.08, right=0.92, top=0.95, bottom=0.05)


    def set_tick_format(ax, cbar, decimals=2):
        formatter = ticker.FormatStrFormatter(f'%.{decimals}f')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        cbar.ax.yaxis.set_major_formatter(formatter)


    def plot_helper_kg(row, col, u_val, title_base, is_error=False):
        ax = plt.subplot(gs[row, col])
        cmap = 'RdBu_r'
        if is_error:
            title = f'Absolute Error,\n{title_base}'
            label = r'$|u_{\text{pred}} - u_{\text{exact}}|$'
        else:
            title = (f'Exact Solution\n(t = {t_final_val})' if title_base == "REF"
                     else f'{title_base}\n(t = {t_final_val})')
            label = r'$u$'

        im = ax.contourf(X_ij, Y_ij, u_val, levels=50, cmap=cmap)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        set_tick_format(ax, cbar, decimals=2)


    # 1열~2열: Exact, SPIKAN, OR, XOR
    plot_helper_kg(0, 0, u_exact_final, "REF")

    if 'spikan' in saved_datasets:
        plot_helper_kg(1, 0, saved_datasets['spikan'][:, :, t_idx_final], "SPIKAN")
        plot_helper_kg(1, 1, np.abs(saved_datasets['spikan'][:, :, t_idx_final] - u_exact_final), "SPIKAN",
                       is_error=True)

    if 'or-spikan' in saved_datasets:
        plot_helper_kg(2, 0, saved_datasets['or-spikan'][:, :, t_idx_final], "OR-SPIKAN")
        plot_helper_kg(2, 1, np.abs(saved_datasets['or-spikan'][:, :, t_idx_final] - u_exact_final), "OR-SPIKAN",
                       is_error=True)

    if 'xor-spikan' in saved_datasets:
        plot_helper_kg(3, 0, saved_datasets['xor-spikan'][:, :, t_idx_final], "XOR-SPIKAN")
        plot_helper_kg(3, 1, np.abs(saved_datasets['xor-spikan'][:, :, t_idx_final] - u_exact_final), "XOR-SPIKAN",
                       is_error=True)

    # 3열~4열: tanh/sigmoid 계열
    other_modes = ['tanh_or-spikan', 'tanh_xor-spikan', 'sigmoid_or-spikan', 'sigmoid_xor-spikan']
    for idx, mode in enumerate(other_modes):
        if mode in saved_datasets:
            u_p = saved_datasets[mode][:, :, t_idx_final]
            # ★ 변경된 부분: title_upper 로직을 format_mode_name으로 교체
            title_upper = format_mode_name(mode)

            plot_helper_kg(idx, 2, u_p, title_upper)
            plot_helper_kg(idx, 3, np.abs(u_p - u_exact_final), title_upper, is_error=True)

    # ★ results_dir에 PNG + PDF 저장 (용량 조절을 원하시면 dpi=600 숫자를 변경)
    final_plot_path = os.path.join(results_dir, 'comparison_kg_4x4_optimized_dpi_150.png')
    plt.savefig(final_plot_path, dpi=150, bbox_inches='tight')
    plt.savefig(final_plot_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()