import pandas as pd
from pathlib import Path
from initial_pipeline.initial_pipeline_utils.data_loader import DataLoader


def load_data(*paths: Path) -> list[pd.DataFrame]:
    """
    Loads and transforms data from one or more file paths.

    Returns a list of DataFrames in the same order as the input paths.
    """
    dataloader = DataLoader()
    dataframes = []

    for path in paths:
        df = dataloader.load_transform(path)
        print(f"Data import and transformation finished for: {path}")
        dataframes.append(df)

    return dataframes
