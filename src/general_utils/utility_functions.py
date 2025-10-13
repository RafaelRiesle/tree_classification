import pandas as pd
from pathlib import Path
from pipelines.preprocessing.preprocessing_pipeline_utils.data_loader import DataLoader


def load_data(*paths: Path):
    """
    Loads and transforms data from one or more file paths.

    Returns a DataFrame if one path is given, otherwise a list of DataFrames.
    """
    dataloader = DataLoader()
    dataframes = []

    for path in paths:
        df = dataloader.load_transform(path)
        print(f"Data import and transformation finished for: {path}")
        dataframes.append(df)

    if len(dataframes) == 1:
        return dataframes[0]
    return dataframes
