
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import pandas as pd
import io

from uav.evaluation.models import FullAnalaysisRegistry
from uav.visualization.utils import load_results
from uav.visualization.cfg import IMPLEMENTATIONS, COLORS, MARKERS, SIG_COLOR, SIG_THRESHOLD


# TODO
# 1. parse together datablock depending on analysis type
# 2. render full analysis plot

def render_full_statistical_analysis(results_filepath: str, full_analysis_png_filepath_base: str, analysis: FullAnalaysisRegistry) -> None:

    png_filepath = f"{full_analysis_png_filepath_base}/{analysis.value}.png"
    df = load_results(results_filepath)


    df = pd.read_csv(io.StringIO(
        """implementation,measured_metric,friedman_p,wilcoxon_p_1v2,wilcoxon_p_2v3,wilcoxon_p_1v3,hommel_p_1v2,hommel_p_2v3,hommel_p_1v3,effect_size
            r,PRECISION,0.2026061711071116,0.8589549227374824,0.1240427930900948,0.1626733150719467,0.8589549227374824,0.2480855861801896,0.3253466301438934
            r,RECALL,0.070713214998585,1.0,0.0735939238114404,0.0979893027079174,1.0,0.1471878476228809,0.1959786054158348
            r,mAP50,0.0572888580930433,0.9527650219907529,0.0735939238114404,0.0626713878689624,0.9527650219907529,0.1471878476228809,0.1253427757379249
            r,mAP50_95,0.013123728736941,0.9527650219907529,0.0555329808159427,0.0626713878689624,0.9527650219907529,0.1110659616318854,0.1253427757379249
        """
    ))

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

    METRICS = df['measured_metric'].unique()
    SIG_THRESHOLD = 0.05
    SIG_COLOR = '#d62728'

    COLOR_MAP = {
        'Friedman': '#555555', 
        'Wilcoxon': '#ff7f0e', 
        'Hommel': '#2ca02c'    
    }

    MARKER_MAP = {
        '1v2': 'o',
        '2v3': 's',
        '1v3': '^',
        'global': 'D' 
    }

    _, ax = plt.subplots(figsize=(12, 7))

    x_indices = np.arange(len(METRICS))
    offset_map = {'Friedman': -0.2, 'Wilcoxon': 0.0, 'Hommel': 0.2}

    ax.axhspan(0, SIG_THRESHOLD, facecolor=SIG_COLOR, alpha=0.1, zorder=0)
    ax.axhline(SIG_THRESHOLD, color=SIG_COLOR, linestyle='--', linewidth=1.5, label=f'Significance (p={SIG_THRESHOLD})')

    for i, metric in enumerate(METRICS):
        row = df[df['measured_metric'] == metric].iloc[0]
        
        f_val = row['friedman_p']
        x_f = x_indices[i] + offset_map['Friedman']
        
        ax.scatter(x_f, f_val, color=COLOR_MAP['Friedman'], marker=MARKER_MAP['global'], 
                s=100, edgecolors='white', zorder=3)
        
        fw = 'bold' if f_val < SIG_THRESHOLD else 'normal'
        ax.text(x_f, f_val + 0.02, f'{f_val:.3f}', ha='center', va='bottom', 
                fontsize=8, fontweight=fw, color=COLOR_MAP['Friedman'])

        test_groups = [
            ('Wilcoxon', ['wilcoxon_p_1v2', 'wilcoxon_p_2v3', 'wilcoxon_p_1v3']),
            ('Hommel',   ['hommel_p_1v2',   'hommel_p_2v3',   'hommel_p_1v3'])
        ]
        
        pair_names = ['1v2', '2v3', '1v3']

        for test_name, cols in test_groups:
            vals = [row[c] for c in cols]
            x_base = x_indices[i] + offset_map[test_name]
            
            ax.vlines(x_base, min(vals), max(vals), color=COLOR_MAP[test_name], alpha=0.3, linewidth=2, zorder=1)
            
            for val, pair in zip(vals, pair_names):
                ax.scatter(x_base, val, color=COLOR_MAP[test_name], marker=MARKER_MAP[pair], 
                        s=80, edgecolors='white', zorder=3)
                
                fw = 'bold' if val < SIG_THRESHOLD else 'normal'
                
                if pair == '1v3': 
                    ha_align = 'right'
                    text_offset_x = -0.04
                elif pair == '2v3': 
                    ha_align = 'left'
                    text_offset_x = 0.04
                else:
                    ha_align = 'left'
                    text_offset_x = 0.04
                
                ax.text(x_base + text_offset_x, val, f'{val:.3f}', 
                        ha=ha_align, va='center', fontsize=7, fontweight=fw, color='black')

    ax.set_title('Full Statistical Analysis in R', fontweight='bold', fontsize=14, pad=15)
    ax.set_ylabel('P-Value')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(METRICS, fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(axis='y', linestyle=':', alpha=0.6)


    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP['Friedman'], label='Friedman', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP['Wilcoxon'], label='Wilcoxon', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MAP['Hommel'], label='Hommel', markersize=8),

        Line2D([0], [0], marker=MARKER_MAP['1v2'], color='gray', label='vz vs ir', markerfacecolor='gray', markersize=8, linestyle='None'),
        Line2D([0], [0], marker=MARKER_MAP['2v3'], color='gray', label='ir vs hy', markerfacecolor='gray', markersize=8, linestyle='None'),
        Line2D([0], [0], marker=MARKER_MAP['1v3'], color='gray', label='vz vs hy', markerfacecolor='gray', markersize=8, linestyle='None'),
    ]

    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
            ncol=2, frameon=False, title="Legend")

    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.savefig(png_filepath, dpi=300)
    plt.close()
