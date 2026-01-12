import pandas as pd


def load_results(results_filepath: str) -> pd.DataFrame:
    """Loads statistical analysis results from CSV file into DataFrame."""
    return pd.read_csv(results_filepath)
