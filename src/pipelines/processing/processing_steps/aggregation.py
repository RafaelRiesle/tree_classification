import pandas as pd


class TimeSeriesAggregate:
    def __init__(self, on=True, freq=1, method="median"):
        self.on = on
        self.freq = freq
        self.method = method

    # TODO: use method instead of mean
    def resample_by_freq(self, group, freq):
        group = group.set_index("time").sort_index()
        resampled = group.resample(f"{freq}W-MON").mean(numeric_only=True)
        resampled = resampled.reset_index()
        resampled["id"] = group["id"].iloc[0]
        return resampled

    def aggregate_timeseries(
        self, df: pd.DataFrame, freq: int = 1, method: str = "median"
    ) -> pd.DataFrame:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"])

        agg_methods = {"mean", "median", "sum", "min", "max"}
        if method not in agg_methods:
            raise ValueError(f"Method must be one of {agg_methods}")

        df_categ = df[["id", "species", "disturbance_year"]].drop_duplicates("id")
        df = df.drop(columns=["species", "disturbance_year"])

        df["year_week"] = df["time"].dt.strftime("%Y-%W")
        min_date = df["time"].min()
        max_date = df["time"].max()

        all_yearweeks = pd.period_range(start=min_date, end=max_date, freq="W-MON")
        all_yearweeks = all_yearweeks.to_timestamp().strftime("%Y-%W")

        df = df.groupby(["id", "year_week"], as_index=False).agg(
            method, numeric_only=True
        )
        ids = df["id"].unique()

        full_index = pd.MultiIndex.from_product(
            [ids, all_yearweeks], names=["id", "year_week"]
        )
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
