import re
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


class SpectralBandPlotter:
    def __init__(self, df):
        self.df = df
        self.band_columns = self._get_band_columns()

    def _get_band_columns(self):
        def band_key(col_name):
            number = re.findall(r"\d+", col_name)
            return int(number[0]) if number else 0

        return sorted(
            (col for col in self.df.columns if col.startswith("b")),
            key=band_key,
        )

    def sample_data(self, n=5000):
        self.df_sampled = (
            self.df.groupby("year", group_keys=False)
            .apply(lambda x: x.sample(n=n) if len(x) > n else x)
            .reset_index(drop=True)
        )
        return self.df_sampled

    def _melt_df(self, df):
        df_melted = pd.melt(
            df[["species", "year"] + self.band_columns],
            id_vars=["species", "year"],
            value_vars=self.band_columns,
            var_name="band",
            value_name="value",
        )
        df_melted["band"] = pd.Categorical(
            df_melted["band"], categories=self.band_columns, ordered=True
        )
        return df_melted

    def plot_per_year(self, sample_size=5000):
        df_sampled = self.sample_data(n=sample_size)
        df_melted = self._melt_df(df_sampled)
        df_melted.sort_values(["year", "band", "species"], inplace=True)

        years = df_melted["year"].unique()
        n_years = len(years)

        fig, axes = plt.subplots(
            nrows=n_years, ncols=1, figsize=(24, 6 * n_years), sharex=True
        )
        if n_years == 1:
            axes = [axes]

        for ax, year in zip(axes, years):
            data_year = df_melted[df_melted["year"] == year]
            sns.boxplot(data=data_year, x="band", y="value", hue="species", ax=ax)
            ax.set_title(f"Spectral Band Ranges per Species – Year: {year}")
            ax.set_xlabel("")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.tick_params(axis="x", rotation=45)

        plt.xlabel("Band")
        plt.tight_layout()
        plt.show()

    def plot_all_years(self, sample_size=50000, showfliers=True):
        df_sampled = self.sample_data(n=sample_size)
        df_melted = self._melt_df(df_sampled)
        df_melted.sort_values(["band", "species", "year"], inplace=True)

        plt.figure(figsize=(24, 8))
        sns.boxplot(
            data=df_melted, x="band", y="value", hue="species", showfliers=showfliers
        )
        plt.title("Spectral Band Ranges per Species – All Years")
        plt.xlabel("Band")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_species_season_distribution(self):
        band_columns = [col for col in self.df.columns if col.startswith("b")]
        df_grouped = (
            self.df.groupby(["species", "season"])[band_columns].mean().reset_index()
        )

        df_melted = pd.melt(
            df_grouped,
            id_vars=["species", "season"],
            value_vars=band_columns,
            var_name="band",
            value_name="mean_value",
        )

        fig = px.line(
            df_melted,
            x="band",
            y="mean_value",
            color="species",
            line_group="season",
            facet_col="season",
            markers=True,
            title="Average Spectral Bands per Season and Species",
        )

        fig.update_layout(xaxis_title="Spectral Band", yaxis_title="Average Value")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.show()

    def plot_spectral_development_over_years(self, addition: str = None):
        if addition is None:
            addition = "Overview"
        band_columns = [col for col in self.df.columns if col.startswith("b")]
        fig = px.line(
            self.df,
            x="doy",
            y=band_columns,
            facet_col="year",
            title=f"Spectral Bands Development Over Years ({addition})",
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.show()
