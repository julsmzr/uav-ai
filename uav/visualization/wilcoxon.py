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
        
        ax.axhspan(0, SIG_THRESHOLD, facecolor=SIG_COLOR, alpha=0.1, zorder=0)
        ax.axhline(SIG_THRESHOLD, color=SIG_COLOR, linestyle='--', linewidth=1.5, alpha=0.8)
        
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

            ax.scatter(valid_x, y_vals, color=COLORS[impl], label=impl, 
                    s=90, marker=MARKERS[impl], edgecolors='white', linewidth=1, zorder=3)

        for x_idx, metric in enumerate(METRICS):            
            for order_idx, impl in enumerate(ordered_impls):
                
                row = df[(df['measured_metric'] == metric) & (df['implementation'] == impl)]
                if row.empty: continue
                
                try:
                    val = row[col_name].values[0]
                except IndexError:
                    continue
                
                x_loc = x_idx + offsets[impl]                
                is_above = (order_idx % 2 == 0)
                
                if is_above:
                    y_loc = val + 0.03
                    va = 'bottom'
                else:
                    y_loc = val - 0.03
                    va = 'top'

                fw = 'bold' if val < SIG_THRESHOLD else 'normal'
                
                ax.text(x_loc, y_loc, f'{val:.3f}', 
                        ha='center', va=va, 
                        fontsize=7, fontweight=fw, color='black', zorder=4)

        ax.set_title(comp_labels[i], fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(METRICS, rotation=45, ha='right')
        
        ax.set_ylim(0.0, 1.05) 
        ax.grid(axis='y', linestyle=':', alpha=0.6, linewidth=1)
        
        if i == 0:
            ax.set_ylabel('Wilcoxon P-Value')

    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = {l: h for l, h in zip(labels, handles)}
    
    fig.legend(unique_labels.values(), unique_labels.keys(), 
                loc='lower center', ncol=3, 
                bbox_to_anchor=(0.5, 0.02), frameon=False, title='Implementation')

    plt.suptitle('Paired Wilcoxon Test Results', fontsize=14, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(bottom=0.20, top=0.90, wspace=0.2, left=0.08, right=0.98)
    plt.savefig(wilcoxon_png_filepath, dpi=300)
    plt.close()
