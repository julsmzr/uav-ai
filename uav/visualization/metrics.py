import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_violin_plots(csv_path: str, output_path: str) -> None:
    """Renders violin plots showing metric distributions for VZ, IR, and HY experiments."""
    df = pd.read_csv(csv_path)
    df['experiment'] = df['experiment_name'].apply(lambda x: x.split('-')[0].upper())

    if 'mAP50-95' in df.columns:
        df = df.rename(columns={'mAP50-95': 'mAP50_95'})

    metrics_map = {
        'precision': 'PRECISION',
        'recall': 'RECALL',
        'mAP50': 'mAP50',
        'mAP50_95': 'mAP50_95'
    }

    actual_columns = [col for col in ['precision', 'recall', 'mAP50', 'mAP50_95'] if col in df.columns]
    df_subset = df[['experiment'] + actual_columns].copy()

    df_subset = df_subset.rename(columns=metrics_map)
    metrics = list(metrics_map.values())

    stats = df_subset.groupby('experiment')[metrics].agg(['mean', 'std'])
    experiment_order = ['VZ', 'IR', 'HY']

    df_melted = df_subset.melt(
        id_vars=['experiment'], 
        value_vars=metrics, 
        var_name='Metric', 
        value_name='Score'
    )

    g = sns.catplot(
        data=df_melted,
        x='experiment',
        y='Score',
        col='Metric',
        kind='violin',
        col_wrap=4,
        order=experiment_order,
        sharey=False,
        sharex=False,
        height=5,
        aspect=0.7,
        hue='experiment',
        palette={'VZ': '#2E86AB', 'IR': '#A23B72', 'HY': '#F18F01'},
        legend=False,
        inner='quartile',
        cut=0
    )

    g.set_titles("{col_name}", fontweight='bold', size=14)

    for ax in g.axes.flatten():
        metric_name = ax.get_title()

        ranking_data = []
        for exp in experiment_order:
            if exp in stats.index:
                m_val = stats.loc[exp, (metric_name, 'mean')]
                s_val = stats.loc[exp, (metric_name, 'std')]
                ranking_data.append({'exp': exp, 'mean': m_val, 'std': s_val})

        ranking_data.sort(key=lambda x: (-x['mean'], x['std']))
        rank_map = {item['exp']: i for i, item in enumerate(ranking_data)}

        new_labels = []
        for exp_name in experiment_order:
            if exp_name in stats.index:
                m = stats.loc[exp_name, (metric_name, 'mean')]
                s = stats.loc[exp_name, (metric_name, 'std')]

                rank = rank_map[exp_name]

                # Generate newlines based on rank
                newlines = "\n" * (rank + 1)
                label_text = f"{exp_name}{newlines}{m:.4f} ± {s:.4f}"
            else:
                label_text = f"{exp_name}\nN/A"

            new_labels.append(label_text)

        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(new_labels, fontweight='bold', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    g.set_axis_labels("", "Score [1]")
    g.fig.suptitle('Performance Distribution', fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    render_violin_plots("results/metrics.csv", "results/visualizations/metrics.png")
