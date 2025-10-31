import pandas as pd


class TimeSeriesAggregate:
    def __init__(self, on=True, freq=2, method="mean"):
        """
        Parameters
        ----------
        on : bool
            Whether to run aggregation.
        freq : int
            Frequency in weeks for resampling.
        method : str
            Aggregation method ('mean', 'median', 'sum', 'min', 'max')
        """
        self.on = on
        self.freq = freq
        self.method = method

    def resample_by_freq(self, group, freq):
        group = group.set_index("time").sort_index()
        resampled = (
            group.resample(f"{freq}W-MON")
            .agg(self.method, numeric_only=True)
            .reset_index()
        )
        resampled["id"] = group["id"].iloc[0]
        return resampled

    def aggregate_timeseries(self, df: pd.DataFrame, freq: int = 1, method: str = "median") -> pd.DataFrame:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        agg_methods = {"mean", "median", "sum", "min", "max"}
        if method not in agg_methods:
            raise ValueError(f"Method must be one of {agg_methods}")

        meta_cols = [c for c in ["id", "species", "disturbance_year"] if c in df.columns]
        df_categ = df[meta_cols].drop_duplicates("id") if len(meta_cols) > 1 else df[["id"]].drop_duplicates("id")
        df = df.drop(columns=[c for c in ["species", "disturbance_year"] if c in df.columns], errors="ignore")

        df["year_week"] = df["time"].dt.strftime("%Y-%W")
        min_date = df["time"].min()
        max_date = df["time"].max()

        all_yearweeks = pd.period_range(start=min_date, end=max_date, freq="W-MON")
        all_yearweeks = all_yearweeks.to_timestamp().strftime("%Y-%W")

        df = df.groupby(["id", "year_week"], as_index=False).agg(method, numeric_only=True)
        ids = df["id"].unique()

        full_index = pd.MultiIndex.from_product([ids, all_yearweeks], names=["id", "year_week"])
        full_df = full_index.to_frame(index=False)
        df = full_df.merge(df, on=["id", "year_week"], how="left")

        df["time"] = pd.to_datetime(df["year_week"] + "-1", format="%Y-%W-%w")

        df = (
            df.groupby("id")
            .apply(lambda g: self.resample_by_freq(g, freq))
            .reset_index(drop=True)
        )
        df = df.merge(df_categ, on="id", how="left")
        df = df.sort_values(["id", "time"]).reset_index(drop=True)
        return df


    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df
        return self.aggregate_timeseries(df, freq=self.freq, method=self.method)
