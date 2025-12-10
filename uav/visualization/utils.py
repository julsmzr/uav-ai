import pandas as pd


def load_results(results_filepath: str) -> pd.DataFrame:
    return pd.read_csv(results_filepath)
