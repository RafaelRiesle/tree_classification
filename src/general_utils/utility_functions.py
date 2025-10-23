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


def get_id_sample(df, id_col="id", time_col="time", n_ids=40):
    """
    Returns a subset of the DataFrame with up to `n_ids` unique IDs.
    """
    df_sorted = df.sort_values(by=[id_col, time_col])
    unique_ids = df_sorted[id_col].drop_duplicates()
    sampled_ids = unique_ids.sample(n=min(n_ids, len(unique_ids)), random_state=42)
    return df_sorted[df_sorted[id_col].isin(sampled_ids)].reset_index(drop=True)
