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
