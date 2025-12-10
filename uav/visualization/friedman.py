import numpy as np
import matplotlib.pyplot as plt

from uav.visualization.utils import load_results
from uav.visualization.cfg import IMPLEMENTATIONS, COLORS, MARKERS, SIG_COLOR, SIG_THRESHOLD


def render_friedman(results_filepath: str, friedman_png_filepath: str) -> None:

    df = load_results(results_filepath)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    METRICS = df['measured_metric'].unique()

    _, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(METRICS))
    width = 0.12

    for i, impl in enumerate(IMPLEMENTATIONS):
        subset = df[df['implementation'] == impl].set_index('measured_metric').reindex(METRICS)
        p_values = subset['friedman_p']
        offset = x + (i * width) - (width * len(IMPLEMENTATIONS) / 2) + (width / 2)
        ax.scatter(offset, p_values, label=impl, color=COLORS[impl], 
                    marker=MARKERS[impl], s=150, edgecolors='white', linewidth=1.5, zorder=3)

    for i, metric in enumerate(METRICS):
        sub = df[df['measured_metric'] == metric]
        val_scipy = sub[sub['implementation'] == 'scipy']['friedman_p'].values[0]
        val_r = sub[sub['implementation'] == 'r']['friedman_p'].values[0]
        val_ping = sub[sub['implementation'] == 'pinguoin']['friedman_p'].values[0]
        
        x_base = i
        x_scipy = x_base - 0.12
        x_r = x_base
        x_ping = x_base + 0.12
        
        ax.plot([x_scipy, x_r], [val_scipy, val_r], color='gray', linewidth=1, alpha=0.5, zorder=1)
        
        center_sr = (x_scipy + x_r) / 2
        fw_sr = 'bold' if val_scipy < SIG_THRESHOLD else 'normal'
        ax.text(center_sr, val_scipy + 0.01, f'{val_scipy:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight=fw_sr)
        
        fw_p = 'bold' if val_ping < SIG_THRESHOLD else 'normal'
        ax.text(x_ping, val_ping + 0.01, f'{val_ping:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight=fw_p)

    ax.axhline(SIG_THRESHOLD, color=SIG_COLOR, linestyle='--', linewidth=1.5, label=f'Significance (p={SIG_THRESHOLD})')
    ax.axhspan(0, SIG_THRESHOLD, facecolor=SIG_COLOR, alpha=0.1, zorder=0)
    ax.set_ylabel('Friedman P-Value')
    ax.set_title('Friedman Test Results', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.grid(axis='y', linestyle=':', alpha=0.6, linewidth=1)
    ax.set_ylim(0, df['friedman_p'].max() + 0.05)
    ax.legend(title='Implementation', frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig(friedman_png_filepath, dpi=300)
    plt.close()
