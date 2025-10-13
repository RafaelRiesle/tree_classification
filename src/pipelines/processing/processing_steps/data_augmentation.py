import pandas as pd
import numpy as np
from tsaug import Drift, AddNoise
from scipy.interpolate import interp1d
import warnings
from preprocessing.preprocessing_pipeline.constants import spectral_bands

warnings.filterwarnings("ignore")


class DataAugmentation:
    def __init__(self, on=True, scale=0.002, drift=0.01):
        self.on = on
        self.scale = scale
        self.drift = drift

    def _make_augmenter(self):
        return Drift(max_drift=self.drift) + AddNoise(scale=self.scale)

    def run(self, df):
        """
        Augments spectral time-series data for all species except Norway spruce.

        Each non-spruce species is upsampled and augmented with small drift and noise 
        using `tsaug`, then linearly interpolated to match the time resolution of the 
        Norway spruce series. The augmented data is combined with the original spruce data.

        Parameters
        ----------
        df : pandas.DataFrame
            Input dataframe with columns: 'species', 'time', and spectral bands.

        Returns
        -------
        pandas.DataFrame
            Augmented dataframe with balanced time-series lengths across species.
        """
        if not self.on:
            return df
        
        df_trees_filter = df[df["species"] != "Norway_spruce"]
        spruce = df[df["species"] == "Norway_spruce"]

        augmented_dfs = []
        max_len = len(spruce)

        for species, df_species in df_trees_filter.groupby("species"):
            df_species["time"] = pd.to_datetime(df_species["time"])
            df_species = df_species.sort_values("time").reset_index(drop=True)

            species_len = len(df_species)
            scale_factor = round(max_len / species_len, 2)
            time = df_species["time"].reset_index(drop=True)

            time_aug = pd.date_range(
                start=time.min(),
                end=time.max(),
                periods=int(len(time) * scale_factor)
            )

            aug_data = {"species": species, "time": time_aug}
            augmenter = self._make_augmenter()

            for col in spectral_bands:
                if len(df_species[col].dropna()) < 5:
                    continue
                Y = df_species[col].interpolate().to_numpy().astype(np.float64)
                Y_aug = augmenter.augment(Y)
                f = interp1d(df_species["time"].view(np.int64), Y_aug, kind="linear", fill_value="extrapolate")
                aug_data[col] = f(time_aug.view(np.int64))

            df_aug_species = pd.DataFrame(aug_data)
            augmented_dfs.append(df_aug_species)

        df_aug_all = pd.concat(augmented_dfs, ignore_index=True)
        df_trees = spruce.drop(columns=["id", "disturbance_year", "doy"], errors="ignore")
        df_aug_all = pd.concat([df_aug_all, df_trees])
        return df_aug_all
