import pandas as pd


class TimeSeriesFilter:
    """
    Removes time series with large median time gaps.
    """

    def __init__(self, time_col="time", id_col="id", max_median_diff_days=18, on=True):
        self.on = on
        self.time_col = time_col
        self.id_col = id_col
        self.max_median_diff_days = max_median_diff_days
        self.valid_ids_ = None

    def fit(self, df: pd.DataFrame):
        # Compute median time difference per ID
        time_diffs = (
            df.groupby(self.id_col)
            .agg({self.time_col: lambda x: x.diff().median()})
            .reset_index()
        )
        time_diffs.columns = [self.id_col, "median_time_diff_days"]
        time_diffs["median_time_diff_days"] = time_diffs[
            "median_time_diff_days"
        ].dt.total_seconds() / (24 * 60 * 60)

        self.valid_ids_ = set(
            time_diffs.loc[
                time_diffs["median_time_diff_days"] <= self.max_median_diff_days,
                self.id_col,
            ]
        )
        return self

    def transform(self, df: pd.DataFrame):
        if self.valid_ids_ is None:
            raise RuntimeError("Call fit() before transform().")
        return df[df[self.id_col].isin(self.valid_ids_)].copy()

    def run(self, df: pd.DataFrame):
        if self.on:
            return self.fit(df).transform(df)
        return df
