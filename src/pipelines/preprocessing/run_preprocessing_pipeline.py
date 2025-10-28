import pandas as pd
from pathlib import Path
from general_utils.constants import spectral_bands
from general_utils.utility_functions import load_data, get_id_sample
from pipelines.preprocessing.preprocessing_pipeline_utils.sits_outlier_cleaner import (
    SITSOutlierCleaner,
)
from pipelines.preprocessing.preprocessing_pipeline_utils.train_test_split import (
    DatasetSplitLoader,
)


BASE_DIR = Path(__file__).parents[3]
DATA_PATH = BASE_DIR / "data/raw/raw_trainset.csv"
SPLITS_PATH = BASE_DIR / "data/raw/splits"
PREPROCESSED_PATH = BASE_DIR / "data/preprocessed"


def create_splits(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int | None = 50,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> dict:
    """Create train/test/validation splits and save them to CSV."""
    if sample_size is not None:
        df = get_id_sample(df, n_ids=sample_size)

    return DatasetSplitLoader(df, output_path=output_path).create_splits(
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        validation_ratio=val_ratio,
        force=True,
    )


def load_or_create_splits(
    df: pd.DataFrame,
    output_path: Path,
    sample_size: int = 50,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    force: bool = False,
) -> dict:
    train_path = output_path / "trainset.csv"
    test_path = output_path / "testset.csv"
    val_path = output_path / "valset.csv"

    if force:
        print("Force flag is True — recreating splits and overwriting old files...")
        output_path.mkdir(parents=True, exist_ok=True)
        splits = create_splits(
            df, output_path, sample_size, train_ratio, test_ratio, val_ratio
        )
        return splits

    if train_path.exists() and test_path.exists() and val_path.exists():
        print("Existing splits found, loading them...")
        splits = {
            "train": pd.read_csv(train_path, parse_dates=["time"]),
            "test": pd.read_csv(test_path, parse_dates=["time"]),
            "val": pd.read_csv(val_path, parse_dates=["time"]),
        }
    else:
        print("Creating new splits...")
        output_path.mkdir(parents=True, exist_ok=True)
        splits = create_splits(
            df, output_path, sample_size, train_ratio, test_ratio, val_ratio
        )

    return splits


def run_outlier_detection(
    splits: dict,
    preprocessed_output: Path,
    contamination: float = 0.05,
    force: bool = False,
):
    """
    Detect and remove outliers from each split and save only interpolated data.
    Only overwrite files if force=True.
    """
    preprocessed_output.mkdir(parents=True, exist_ok=True)
    cleaner = SITSOutlierCleaner(contamination=contamination)
    for name, df_split in splits.items():
        output_file = preprocessed_output / f"{name}set.csv"
        if output_file.exists() and not force:
            print(f"{output_file} exists and force=False, skipping...")
            continue
        cleaner.fit_transform(df_split, spectral_bands)
        interpolated_df = cleaner.get_interpolated_only()
        interpolated_df.to_csv(output_file, index=False)
        print(f"{name} split cleaned and saved at {output_file}")


def run_preprocessing_pipeline(
    data_path: Path = DATA_PATH,
    splits_output_path: Path = SPLITS_PATH,
    preprocessed_output_path: Path = PREPROCESSED_PATH,
    sample_size: int = 2000,
    train_ratio: float = 0.7,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    remove_outliers: bool = False,
    contamination: float = 0.01,
    force_split_creation: bool = False,
    years: list[int] | None = None,
    force_preprocessing=True,
):
    """Run the full preprocessing pipeline with configurable options."""
    preprocessed_files_exist = all(
        (PREPROCESSED_PATH / f"{split}set.csv").exists()
        for split in ["train", "test", "val"]
    )

    if preprocessed_files_exist and not force_preprocessing:
        print("[1] Skipping preprocessing — existing files found.")
        return

    print("=== Starting Preprocessing Pipeline ===")
    df = load_data(data_path)
    if years is not None:
        df["time"] = pd.to_datetime(df["time"])
        df = df[df["time"].dt.year.isin(years)]

    splits = load_or_create_splits(
        df,
        splits_output_path,
        sample_size=sample_size,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        force=force_split_creation,
    )

    preprocessed_output_path.mkdir(parents=True, exist_ok=True)

    if remove_outliers:
        print("[Preprocessing] Running outlier detection...")
        run_outlier_detection(
            splits,
            preprocessed_output_path,
            contamination=contamination,
            force=force_split_creation,
        )
    else:
        print(
            "[Preprocessing] Skipping outlier detection — copying splits to preprocessed folder."
        )
        for split_name, df_split in splits.items():
            output_file = preprocessed_output_path / f"{split_name}set.csv"
            df_split.to_csv(output_file, index=False)
            print(f"✓ Saved {split_name} split to {output_file}")
    print("=== Preprocessing Finished ===")


if __name__ == "__main__":
    run_preprocessing_pipeline(
        data_path=DATA_PATH,
        splits_output_path=SPLITS_PATH,
        preprocessed_output_path=PREPROCESSED_PATH,
        sample_size=None,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        remove_outliers=False,
        contamination=0.05,
        force_split_creation=True,
        # years=[2018, 2019, 2020],
    )
