import pandas as pd
from pathlib import Path
from general_utils.constants import spectral_bands
from general_utils.utility_functions import load_data
from initial_pipeline.initial_pipeline_utils.train_test_split import DatasetSplitLoader
from initial_pipeline.initial_pipeline_utils.sits_outlier_cleaner import (
    SITSOutlierCleaner,
)


DATA_PATH = Path("../../data/raw/raw_trainset.csv")
OUTPUT_PATH = Path("../../data/raw/splits")
PROCESSED_OUTPUT_PATH = Path("../../data/processed")


# TODO Entfernen von sample size
def create_splits(df: pd.DataFrame, output_path: Path, sample_size: int = 300) -> dict:
    dataset_splitter = DatasetSplitLoader(
        df.sample(sample_size), output_path=output_path
    )
    return dataset_splitter.create_splits(
        train_ratio=0.7, test_ratio=0.2, validation_ratio=0.1, force=True
    )


def run_outlier_detection(
    splits, processed_output=PROCESSED_OUTPUT_PATH, contamination=0.05
):
    processed_output.mkdir(parents=True, exist_ok=True)

    cleaner = SITSOutlierCleaner(contamination=contamination)
    processed_files = {}

    for name, df_split in splits.items():
        print(f"Cleaning split: {name}")
        cleaned_df = cleaner.fit_transform(df_split, spectral_bands)

        output_path = processed_output / f"{name}.csv"
        cleaned_df.to_csv(output_path, index=False)

        processed_files[name] = output_path
        print(f"Cleaned dataset saved at: {output_path}")
    print("All splits successfully cleaned and saved.")


def run_initial_pipeline(
    data_path=DATA_PATH,
    output_path=OUTPUT_PATH,
    processed_output=PROCESSED_OUTPUT_PATH,
    contamination=0.05,
):
    df = load_data(data_path)
    splits = create_splits(df, output_path)
    run_outlier_detection(
        splits, processed_output=processed_output, contamination=contamination
    )


if __name__ == "__main__":
    run_initial_pipeline(contamination=0.05)
