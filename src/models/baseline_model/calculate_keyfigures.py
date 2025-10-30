import pandas as pd


class StatisticalFeatures:
    def calculate_keyfigures_per_id(self, df, bands_and_indices):
        """
        Aggregates features per 'id' by month and computes key statistics:
        mean, std, min, max for each feature in bands_and_indices
        """
        df = df.copy()
        df["month"] = df["time"].dt.month_name()

        monthly_agg = df.groupby(["id", "month"])[bands_and_indices].agg(
            ["mean", "std", "min", "max"]
        )

        monthly_agg.columns = [
            "_".join([col[0], col[1]]) for col in monthly_agg.columns.values
        ]
        monthly_agg = monthly_agg.reset_index()

        df_train = monthly_agg.pivot(
            index="id", columns="month", values=monthly_agg.columns[1:]
        )

        df_train.columns = ["_".join(col).strip() for col in df_train.columns.values]
        df_train = df_train.reset_index()

        labels = df.groupby("id")["species"].first().reset_index()
        df_train = df_train.merge(labels, on="id", how="left")

        df_train = df_train.drop(
            columns=[col for col in df_train.columns if "month_" in col]
        )

        # dtypes
        feature_cols = df_train.columns.drop("species")
        for col in feature_cols:
            df_train[col] = pd.to_numeric(df_train[col], errors="coerce")

        return df_train
