

from uav.evaluation.models import FullAnalaysisRegistry
from uav.visualization.friedman import render_friedman
from uav.visualization.wilcoxon import render_wilcoxon
from uav.visualization.full_statistical_analysis import render_full_statistical_analysis
from uav.visualization.visualization import render_violin_plots


def run_all(
    results_filepath: str,
    metrics_csv_filepath: str,
    friedman_png_filepath: str,
    wilcoxon_png_filepath: str,
    metrics_violin_plots_png_filepath: str,
    full_analysis_png_filepath_base: str,
    full_analyses: list[FullAnalaysisRegistry]
):
    """Run all visualization renderers."""
    render_friedman(results_filepath, friedman_png_filepath)
    render_wilcoxon(results_filepath, wilcoxon_png_filepath)
    render_violin_plots(metrics_csv_filepath, metrics_violin_plots_png_filepath)

    for analysis_implementation in full_analyses:
        render_full_statistical_analysis(results_filepath, full_analysis_png_filepath_base, analysis_implementation)