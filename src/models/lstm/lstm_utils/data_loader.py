import pandas as pd


class CSVDataLoader:
    def load_transform(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Ensure required columns
        if not {"time", "id"}.issubset(df.columns):
            raise ValueError("CSV must contain 'time' and 'id' columns.")

        # Convert time column
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        if df["time"].isna().any():
            raise ValueError("Some 'time' values could not be converted to datetime.")

        # Aggregate non-id/time columns
        agg = {
            col: "mean" if pd.api.types.is_numeric_dtype(dtype) else "first"
            for col, dtype in df.dtypes.items()
            if col not in ["time", "id"]
        }

        # Group, aggregate, sort, and drop duplicates
        return (
            df.groupby(["time", "id"], as_index=False)
            .agg(agg)
            .sort_values(["id", "time"])
            .drop_duplicates()
            .reset_index(drop=True)
        )
