import numpy as np
import matplotlib.pyplot as plt

from uav.visualization.utils import load_results
from uav.visualization.cfg import IMPLEMENTATIONS, COLORS, MARKERS, SIG_COLOR, SIG_THRESHOLD


def render_friedman(results_filepath: str, friedman_png_filepath: str) -> None:
    """Renders Friedman test p-values visualization with significance threshold."""
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
        values = {impl: sub[sub['implementation'] == impl]['friedman_p'].values[0]
                  for impl in IMPLEMENTATIONS}

        x_base = i
        x_positions = {
            'scipy': x_base - 0.12,
            'r': x_base,
            'pinguoin': x_base + 0.12,
        }

        grouped = []
        used = set()
        for impl in IMPLEMENTATIONS:
            if impl in used:
                continue
            group = [impl]
            used.add(impl)
            for other in IMPLEMENTATIONS:
                if other not in used and np.isclose(values[impl], values[other], rtol=1e-9):
                    group.append(other)
                    used.add(other)
            grouped.append(group)

        for group in grouped:
            val = values[group[0]]
            x_coords = [x_positions[impl] for impl in group]

            if len(group) > 1:
                ax.plot([min(x_coords), max(x_coords)], [val, val],
                        color='gray', linewidth=1, alpha=0.5, zorder=1)

            center_x = np.mean(x_coords)
            fw = 'bold' if val < SIG_THRESHOLD else 'normal'
            ax.text(center_x, val + 0.005, f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight=fw)

    ax.axhline(SIG_THRESHOLD, color=SIG_COLOR, linestyle='--', linewidth=1.5, label=f'Significance (p={SIG_THRESHOLD})')
    ax.axhspan(0, SIG_THRESHOLD, facecolor=SIG_COLOR, alpha=0.1, zorder=0)
    ax.set_ylabel('Friedman P-Value [1]')
    ax.set_title('Friedman Test Results', fontweight='bold', pad=15)
    ax.set_ylim(bottom=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.grid(axis='y', linestyle=':', alpha=0.6, linewidth=1)
    ax.legend(title='Implementation', frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig(friedman_png_filepath, dpi=300)
    plt.close()
