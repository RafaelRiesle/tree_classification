import pandas as pd
import hashlib
from pathlib import Path


class DatasetSplitLoader:
    def __init__(self, data_path: str, output_dir: str):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.df = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_and_prepare(self) -> pd.DataFrame:
        """Load CSV and parse 'time' column as datetime."""
        self.df = pd.read_csv(self.data_path)

        if "time" not in self.df.columns or "id" not in self.df.columns:
            raise ValueError("'time' and/or 'id' columns are missing in the dataset.")

        self.df["time"] = pd.to_datetime(self.df["time"], format="%Y-%m-%d")
        return self.df

    def create_splits(
        self, train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1, force=False
    ):
        """
        Split unique 'id's reproducibly into train, test, and validation sets.
        Saves CSVs in output_dir.
        If files already exist and force=False, load them instead of creating new ones.
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load_and_prepare() first.")

        # Dateipfade definieren
        split_files = {
            "train": self.output_dir / "trainset.csv",
            "test": self.output_dir / "testset.csv",
            "val": self.output_dir / "valset.csv",
        }

        # Falls schon existieren und force=False → laden statt neu splitten
        if all(path.exists() for path in split_files.values()) and not force:
            print("Splits already exist – loading from disk")
            return {
                name: pd.read_csv(path, parse_dates=["time"])
                for name, path in split_files.items()
            }

        # --- ansonsten: neu erzeugen ---
        total = train_ratio + test_ratio + validation_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, but got {total}")

        # IDs deterministisch hashen
        unique_ids = sorted(self.df["id"].unique())
        hashed_ids = sorted(
            unique_ids, key=lambda x: hashlib.md5(str(x).encode()).hexdigest()
        )

        n_total = len(hashed_ids)
        n_train = round(n_total * train_ratio)
        n_test = round(n_total * test_ratio)
        n_val = n_total - n_train - n_test

        train_ids = set(hashed_ids[:n_train])
        test_ids = set(hashed_ids[n_train : n_train + n_test])
        val_ids = set(hashed_ids[n_train + n_test :])

        splits = {
            "train": self.df[self.df["id"].isin(train_ids)],
            "test": self.df[self.df["id"].isin(test_ids)],
            "val": self.df[self.df["id"].isin(val_ids)],
        }

        # Speichern
        for name, df_split in splits.items():
            df_split.to_csv(split_files[name], index=False)

        print("New splits created and saved")
        return splits
