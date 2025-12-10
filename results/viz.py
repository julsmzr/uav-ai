import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def render_violin_plots(csv_path: str, output_path: str):
    # 1. Load Data
    df = pd.read_csv(csv_path)

    # 2. Preprocessing
    df['experiment'] = df['experiment_name'].apply(lambda x: x.split('-')[0].upper())

    if 'mAP50-95' in df.columns:
        df = df.rename(columns={'mAP50-95': 'mAP50_95'})

    # 3. Prepare for Plotting
    metrics = ['precision', 'recall', 'mAP50', 'mAP50_95']
    df_subset = df[['experiment'] + metrics]

    # Calculate Mean and Std
    stats = df_subset.groupby('experiment')[metrics].agg(['mean', 'std'])
    
    # HARDCODED ORDER
    experiment_order = ['VZ', 'IR', 'HY']

    df_melted = df_subset.melt(
        id_vars=['experiment'], 
        value_vars=metrics, 
        var_name='Metric', 
        value_name='Score'
    )

    # 4. Visualization
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
        palette={'VZ': '#2E86AB', 'IR': '#A23B72', 'HY': '#F18F01'},
        inner='quartile',
        cut=0
    )

    g.set(ylim=(0.35, 1.05))
    g.set_titles("{col_name}", fontweight='bold', size=14)

    for ax in g.axes.flatten():
        metric_name = ax.get_title()
        
        # REMOVED STRIPPLOT (The dots are gone)
        
       # --- NEW: Calculate Ranks for this Metric ---
        # 1. Collect data for sorting
        ranking_data = []
        for exp in experiment_order:
            if exp in stats.index:
                m_val = stats.loc[exp, (metric_name, 'mean')]
                s_val = stats.loc[exp, (metric_name, 'std')]
                ranking_data.append({'exp': exp, 'mean': m_val, 'std': s_val})
        
        # 2. Sort: Higher Mean wins (-x['mean']), then Lower Std wins (x['std'])
        ranking_data.sort(key=lambda x: (-x['mean'], x['std']))
        
        # 3. Create a lookup map: {'VZ': 0, 'IR': 1, 'HY': 2} (0 is best)
        rank_map = {item['exp']: i for i, item in enumerate(ranking_data)}
        # --------------------------------------------

        new_labels = []
        for exp_name in experiment_order:
            if exp_name in stats.index:
                m = stats.loc[exp_name, (metric_name, 'mean')]
                s = stats.loc[exp_name, (metric_name, 'std')]
                
                # Determine Rank (0 = 1st place, 1 = 2nd place, etc.)
                rank = rank_map[exp_name]
                
                # Generate newlines based on rank
                # Rank 0 (1st) -> "\n"
                # Rank 1 (2nd) -> "\n\n"
                # Rank 2 (3rd) -> "\n\n\n"
                newlines = "\n" * (rank + 1)
                
                label_text = f"{exp_name}{newlines}{m:.4f} ± {s:.4f}"
            else:
                label_text = f"{exp_name}\nN/A"
            
            new_labels.append(label_text)
        
        # SMALLER FONT SIZE (9)
        ax.set_xticklabels(new_labels, fontweight='bold', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # REVERTED TITLE
    g.set_axis_labels("", "Score")
    g.fig.suptitle('Performance Distribution', fontsize=16, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # RENAMED OUTPUT FILE
    render_violin_plots("results/metrics.csv", "results/visualizations/violin_plots.png")