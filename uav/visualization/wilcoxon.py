import numpy as np
import matplotlib.pyplot as plt

from uav.visualization.utils import load_results
from uav.visualization.cfg import IMPLEMENTATIONS, COLORS, MARKERS, SIG_COLOR, SIG_THRESHOLD

def render_wilcoxon(results_filepath: str, wilcoxon_png_filepath: str) -> None:
    """Renders 3-panel Wilcoxon pairwise comparison p-values visualization."""
    df = load_results(results_filepath)

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    METRICS = df['measured_metric'].unique()
    
    comparisons = ['wilcoxon_p_1v2', 'wilcoxon_p_2v3', 'wilcoxon_p_1v3']
    comp_labels = ['vz vs ir', 'ir vs hy', 'vz vs hy']

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes = axes.flatten()

    x_pos = np.arange(len(METRICS))

    offsets = {'scipy': -0.15, 'r': 0.0, 'pinguoin': 0.15}
    ordered_impls = ['scipy', 'r', 'pinguoin']

    for i, col_name in enumerate(comparisons):
        ax = axes[i]

        all_y_vals = []
        for impl in IMPLEMENTATIONS:
            y_vals = []
            valid_x = []
            
            for m_idx, metric in enumerate(METRICS):
                row = df[(df['measured_metric'] == metric) & (df['implementation'] == impl)]
                if not row.empty:
                    val = row[col_name].values[0]
                    y_vals.append(val)
                    valid_x.append(x_pos[m_idx] + offsets[impl])
            
            if not y_vals: continue

            all_y_vals.extend(y_vals)
            scatter = ax.scatter(valid_x, y_vals, color=COLORS[impl], label=impl,
                    s=90, marker=MARKERS[impl], edgecolors='white', linewidth=1, zorder=3)
            scatter.set_clip_on(True)

        if all_y_vals:
            y_range = max(all_y_vals) - min(all_y_vals)
            text_offset = y_range * 0.03 if y_range > 0 else 0.01
            y_min, y_max = min(all_y_vals), max(all_y_vals)
            margin = y_range * 0.15 if y_range > 0 else 0.1
            plot_y_min = max(0, y_min - margin)
            plot_y_max = y_max + margin if y_max >= 0.98 else min(1.0, y_max + margin)
        else:
            text_offset = 0.03
            plot_y_min, plot_y_max = 0, 1.0

        for x_idx, metric in enumerate(METRICS):
            metric_values = {}
            for impl in ordered_impls:
                row = df[(df['measured_metric'] == metric) & (df['implementation'] == impl)]
                if not row.empty:
                    try:
                        val = row[col_name].values[0]
                        metric_values[impl] = val
                    except IndexError:
                        pass

            at_one = [impl for impl, val in metric_values.items() if val == 1.0]

            if len(at_one) > 1:
                x_positions = [x_idx + offsets[impl] for impl in at_one]
                ax.plot(x_positions, [1.0] * len(at_one), color='gray', linewidth=1, alpha=0.5, zorder=1)

                center_x = sum(x_positions) / len(x_positions)
                y_loc = 1.0 - text_offset
                ax.text(center_x, y_loc, '1.000',
                        ha='center', va='top',
                        fontsize=7, fontweight='normal', color='black', zorder=4,
                        clip_on=False)

            for order_idx, impl in enumerate(ordered_impls):
                if impl not in metric_values:
                    continue

                val = metric_values[impl]

                if val == 1.0 and len(at_one) > 1:
                    continue

                x_loc = x_idx + offsets[impl]
                is_above = (order_idx % 2 == 0)

                if is_above:
                    y_loc = val + text_offset
                    va = 'bottom'
                else:
                    y_loc = val - text_offset
                    va = 'top'

                y_loc = max(plot_y_min, min(plot_y_max, y_loc))

                fw = 'bold' if val < SIG_THRESHOLD else 'normal'

                ax.text(x_loc, y_loc, f'{val:.3f}',
                        ha='center', va=va,
                        fontsize=7, fontweight=fw, color='black', zorder=4)

        if all_y_vals:
            if plot_y_min <= SIG_THRESHOLD <= plot_y_max:
                ax.axhline(SIG_THRESHOLD, color=SIG_COLOR, linestyle='--', linewidth=1.5, label=f'Significance (p={SIG_THRESHOLD})')
                span_bottom = max(0, plot_y_min)
                ax.axhspan(span_bottom, SIG_THRESHOLD, facecolor=SIG_COLOR, alpha=0.1, zorder=0)

            ax.set_ylim(plot_y_min, plot_y_max)

        ax.set_title(comp_labels[i], fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(METRICS, rotation=45, ha='right')
        
        ax.grid(axis='y', linestyle=':', alpha=0.6, linewidth=1)
        
        if i == 0:
            ax.set_ylabel('Wilcoxon P-Value [1]')

    all_handles = []
    all_labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    unique_labels = {l: h for l, h in zip(all_labels, all_handles)}

    impl_handles = [unique_labels[impl] for impl in IMPLEMENTATIONS if impl in unique_labels]
    impl_labels = [impl for impl in IMPLEMENTATIONS if impl in unique_labels]

    sig_handle = unique_labels.get(f'Significance (p={SIG_THRESHOLD})', None)

    if sig_handle:
        legend1 = fig.legend(impl_handles, impl_labels,
                            loc='lower center', ncol=3,
                            bbox_to_anchor=(0.5, 0.06), frameon=False, title='Implementation')
        fig.add_artist(legend1)
        fig.legend([sig_handle], [f'Significance (p={SIG_THRESHOLD})'],
                   loc='lower center', ncol=1,
                   bbox_to_anchor=(0.5, 0.02), frameon=False)
    else:
        fig.legend(impl_handles, impl_labels,
                   loc='lower center', ncol=3,
                   bbox_to_anchor=(0.5, 0.04), frameon=False, title='Implementation')

    plt.suptitle('Paired Wilcoxon Test Results', fontsize=14, fontweight='bold', y=0.98)

    plt.subplots_adjust(bottom=0.25, top=0.90, wspace=0.2, left=0.08, right=0.98)
    plt.savefig(wilcoxon_png_filepath, dpi=300)
    plt.close()
