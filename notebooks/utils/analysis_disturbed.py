import pandas as pd
import matplotlib.pyplot as plt


class DisturbedYearAnalysis:
    def __init__(self, df, figsize=(10, 6), color="blue"):
        self.df = df
        self.figsize = figsize
        self.color = color

    def summary_unique_years(self, column="disturbance_year"):
        """Print the number of unique values and the list of unique disturbance years."""
        unique_values = self.df[column].unique()
        print(f"Number of unique values: {len(unique_values)}")
        print(f"Unique values:\n{unique_values}")

    def plot_disturbed_counts(self, column="is_disturbed"):
        """Plot a bar chart showing counts of disturbed vs. non-disturbed entries."""
        counts_df = self.df[column].value_counts().reset_index()
        counts_df.columns = [column, "count"]

        plt.figure(figsize=self.figsize)
        plt.bar(counts_df[column].astype(str), counts_df["count"], color=self.color)
        plt.xlabel(column.replace("_", " ").title())
        plt.ylabel("Count")
        plt.title(f"Comparison of {column.replace('_', ' ').title()} Values")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_disturbance_by_species(
        self, year_column="disturbance_year", species_column="species"
    ):
        """Plot stacked bar chart of disturbance year distribution by species."""
        filtered = self.df[self.df[year_column] != 0]
        crosstab = pd.crosstab(filtered[year_column], filtered[species_column])
        crosstab.sort_index(inplace=True)
        crosstab.plot(kind="bar", stacked=True, figsize=self.figsize, colormap="tab20b")

        plt.xlabel(year_column.replace("_", " ").title())
        plt.ylabel("Count")
        plt.title(
            f"Distribution of {year_column.replace('_', ' ').title()} by {species_column.title()}"
        )
        plt.legend(
            title=species_column.title(), bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()
