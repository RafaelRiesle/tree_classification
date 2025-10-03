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

    def plot_per_year(self, sample_size=5000, showfliers=True):
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
            sns.boxplot(
                data=data_year,
                x="band",
                y="value",
                hue="species",
                ax=ax,
                showfliers=showfliers,
            )
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
        # Identify band columns
        band_columns = [col for col in self.df.columns if col.startswith("b")]

        # Compute mean per species and season
        df_grouped = (
            self.df.groupby(["species", "season"])[band_columns].mean().reset_index()
        )

        # Get unique seasons and species
        seasons = df_grouped["season"].unique()
        species_list = df_grouped["species"].unique()

        # Create subplots: one column per season
        fig, axes = plt.subplots(
            1, len(seasons), figsize=(5 * len(seasons), 5), sharey=True
        )

        if len(seasons) == 1:
            axes = [axes]  # ensure axes is iterable

        for ax, season in zip(axes, seasons):
            df_season = df_grouped[df_grouped["season"] == season]
            for species in species_list:
                df_species = df_season[df_season["species"] == species]
                ax.plot(
                    band_columns,
                    df_species[band_columns].values.flatten(),
                    marker="o",
                    label=species,
                )

            ax.set_title(f"Season: {season}")
            ax.set_xlabel("Spectral Band")
            ax.set_ylabel("Average Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(title="Species")
            ax.set_xticks(range(len(band_columns)))
            ax.set_xticklabels(band_columns, rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_spectral_development_over_years(self, addition: str = None):
        if addition is None:
            addition = "Overview"

        # Identify spectral band columns
        band_columns = [col for col in self.df.columns if col.startswith("b")]

        # Unique years
        years = self.df["year"].unique()
        n_years = len(years)

        # Create subplots (1 row, n_years columns)
        fig, axes = plt.subplots(1, n_years, figsize=(5 * n_years, 5), sharey=True)

        if n_years == 1:
            axes = [axes]  # ensure axes is iterable

        for ax, year in zip(axes, years):
            df_year = self.df[self.df["year"] == year]
            for band in band_columns:
                ax.plot(df_year["time"], df_year[band], label=band, marker=".")
            ax.set_title(f"Year: {year}")
            ax.set_xlabel("Day of Year")
            ax.set_ylabel("Spectral Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8)

        plt.suptitle(f"Spectral Bands Development Over Years ({addition})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
