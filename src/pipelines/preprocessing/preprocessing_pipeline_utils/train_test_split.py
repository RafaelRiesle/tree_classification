import hashlib
import pandas as pd
from pathlib import Path


class DatasetSplitLoader:
    def __init__(self, df: pd.DataFrame, output_path: str):
        self.output_path = Path(output_path)
        self.df = df
        self.output_path.mkdir(parents=True, exist_ok=True)

    def create_splits(
        self, train_ratio=0.7, test_ratio=0.2, validation_ratio=0.1, force=False
    ):
        """
        Split unique 'id's reproducibly into train, test, and validation sets.
        Saves CSVs in output_path.
        If files already exist and force=False, load them instead of creating new ones.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_and_prepare() first.")

        split_files = {
            "train": self.output_path / "trainset.csv",
            "test": self.output_path / "testset.csv",
            "val": self.output_path / "valset.csv",
        }

        if all(path.exists() for path in split_files.values()) and not force:
            print("Splits already exist – loading from disk")
            return {
                name: pd.read_csv(path, parse_dates=["time"])
                for name, path in split_files.items()
            }

        total = train_ratio + test_ratio + validation_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, but got {total}")

        unique_ids = sorted(self.df["id"].unique())
        hashed_ids = sorted(
            unique_ids, key=lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )

        n_total = len(hashed_ids)
        n_train = round(n_total * train_ratio)
        n_test = round(n_total * test_ratio)

        train_ids = set(hashed_ids[:n_train])
        test_ids = set(hashed_ids[n_train : n_train + n_test])
        val_ids = set(hashed_ids[n_train + n_test :])

        splits = {
            "train": self.df[self.df["id"].isin(train_ids)],
            "test": self.df[self.df["id"].isin(test_ids)],
            "val": self.df[self.df["id"].isin(val_ids)],
        }

        for name, df_split in splits.items():
            df_split.to_csv(split_files[name], index=False)

        print("New splits created and saved")
        return splits
