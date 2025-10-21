import pandas as pd


class InterpolateNaNs:
    def __init__(self, on=True, method="linear", threshold=150):
        self.on = on
        self.method = method
        self.threshold = threshold

    def interpolate_group(self, group: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = group.select_dtypes(include="number").columns
        group[numeric_cols] = group[numeric_cols].interpolate(method=self.method)
        return group

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.on:
            return df

        result = (
            df.groupby("id", group_keys=False)
            .apply(self.interpolate_group)
            .reset_index(drop=True)
        )
        return result
