import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class CorrelationAnalysis:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_correlation_matrix(self) -> pd.DataFrame:
        return self.df.corr(min_periods=5, numeric_only=True).round(4)

    def get_top_correlations(self, top_n: int = 5) -> pd.DataFrame:
        corr = self.get_correlation_matrix()
        corr_unstacked = corr.unstack()

        corr_unstacked = corr_unstacked[
            corr_unstacked.index.get_level_values(0)
            != corr_unstacked.index.get_level_values(1)
        ]
        corr_unstacked.index = [tuple(sorted(p)) for p in corr_unstacked.index]
        corr_unstacked = corr_unstacked[~corr_unstacked.index.duplicated()]

        corr_df = pd.DataFrame(
            {"Columns": corr_unstacked.index, "Value": corr_unstacked.values.round(4)}
        )

        corr_df["Columns"] = corr_df["Columns"].apply(lambda x: f"{x[0]} - {x[1]}")

        pos = corr_df.nlargest(top_n, "Value")
        neg = corr_df.nsmallest(top_n, "Value")

        return pd.concat([pos, neg], ignore_index=True)

    def plot_correlation_matrix(self):
        corr = self.df.corr(numeric_only=True, method="spearman")
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            center=0,
            linewidths=0.5,
            square=True,
            cmap="coolwarm",
            fmt=".1f",
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.show()

    def get_correlations_with_target(self, target_col: str):
        corr_series = self.df.corr(numeric_only=True)[target_col].drop(target_col)
        corr_series = corr_series.sort_values(ascending=False)

        df_corr = pd.DataFrame(
            {"feature": corr_series.index, "correlation": corr_series.values}
        )

        df_corr["abs_corr"] = df_corr["correlation"].abs()

        return df_corr

    def plot_correlations_with_target(self, df_corr, top_n: int):
        df_corr = df_corr.sort_values(by="abs_corr", ascending=True).tail(top_n)
        df_corr = df_corr.reset_index(drop=True)
        colors = [
            "orange" if val < 0 else "royalblue" for val in df_corr["correlation"]
        ]

        plt.figure(figsize=(10, 7))
        plt.barh(
            df_corr["feature"],
            df_corr["abs_corr"],
            color=colors,
        )

        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Correlations\n(orange = negative, blue = positive)")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.show()
