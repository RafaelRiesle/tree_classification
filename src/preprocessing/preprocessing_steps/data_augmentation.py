from tsaug import Drift, AddNoise
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import warnings
from constants import spectral_bands
warnings.filterwarnings("ignore")


def make_augmenter(series):
    scale = 0.002 
    drift = 0.01                   
    return Drift(max_drift=drift) + AddNoise(scale=scale)


def augment(df):
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
     
    df_trees_filter = df[df["species"]!="Norway_spruce"]
    spurce = df[df["species"]=="Norway_spruce"]

    augmented_dfs = []

    max_len = len(spurce)

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

        for col in spectral_bands:
            if len(df_species[col].dropna()) < 5:
                continue
            Y = df_species[col].interpolate().to_numpy().astype(np.float64)

            augmenter = make_augmenter(df_species[col])
            Y_aug = augmenter.augment(Y)

            
            f = interp1d(df_species["time"].view(np.int64), Y_aug, kind="linear", fill_value="extrapolate")
            Y_aug_resampled = f(time_aug.view(np.int64))
            aug_data[col] = Y_aug_resampled

        df_aug_species = pd.DataFrame(aug_data)
        augmented_dfs.append(df_aug_species)
        
    df_aug_all = pd.concat(augmented_dfs, ignore_index=True)
    df_trees = spurce.drop(columns=["id", "disturbance_year", "doy"])
    df_aug_all = pd.concat([df_aug_all, df_trees])
    return df_aug_all