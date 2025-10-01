import matplotlib.pyplot as plt


class HistogramDataVisualization:
    def __init__(
        self,
        df,
        figsize=(10, 6),
        color="black",
        title_fontsize=16,
        label_fontsize=12,
        rotation=45,
    ):
        self.df = df
        self.figsize = figsize
        self.color = color
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.rotation = rotation

    def plot_unique_ids(self, column):
        # Anzahl unique IDs pro Jahr berechnen
        counts = self.df.groupby(column)["id"].nunique().reset_index(name="unique_ids")

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(counts[column], counts["unique_ids"], color=self.color)
        ax.bar_label(bars)

        ax.set_title(f"Unique IDs per {column}", fontsize=self.title_fontsize)
        ax.set_xlabel(column, fontsize=self.label_fontsize)
        ax.set_ylabel("Unique IDs", fontsize=self.label_fontsize)
        plt.xticks(rotation=self.rotation, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_median_id_distribution(self):
        result = self.df.groupby(["id", "year"]).size().reset_index(name="count")
        median_per_year = (
            result.groupby("year")["count"].median().reset_index(name="median_count")
        )

        fig, ax = plt.subplots(figsize=self.figsize)  # gleiche figsize
        bars = ax.bar(
            median_per_year["year"], median_per_year["median_count"], color=self.color
        )
        ax.bar_label(bars)

        ax.set_title("Median Count per Year (ID)", fontsize=self.title_fontsize)
        ax.set_xlabel("Year", fontsize=self.label_fontsize)
        ax.set_ylabel("Median Count", fontsize=self.label_fontsize)
        ax.set_xticks(median_per_year["year"])
        plt.xticks(rotation=self.rotation, ha="right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
