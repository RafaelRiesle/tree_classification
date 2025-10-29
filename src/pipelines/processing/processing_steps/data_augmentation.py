import pandas as pd
import numpy as np
from tsaug import Drift, AddNoise
from scipy.interpolate import interp1d
import warnings
from general_utils.constants import spectral_bands
import tqdm

warnings.filterwarnings("ignore")


class DataAugmentation:
    def __init__(self, on=True, scale=0.002, drift=0.01, threshold=150):
        self.on = on
        self.scale = scale
        self.drift = drift
        self.threshold = threshold

    def _make_augmenter(self):
        return Drift(max_drift=self.drift) + AddNoise(scale=self.scale)

    def resample_time_series(
        self, df_species, spectral_bands, max_len, augmenter, start_time
    ):
        df_species = df_species.sort_values("time").reset_index(drop=True)
        time_aug = pd.date_range(start_time, df_species["time"].max(), periods=max_len)

        aug_data = {
            "species": df_species["species"].iloc[0],
            "id": df_species["id"].iloc[0],
            "time": time_aug,
        }

        for col in spectral_bands:
            Y = df_species[col].interpolate().to_numpy().astype(np.float64)
            if len(Y) < 2:
                aug_data[col] = np.full(max_len, np.nan)
                continue

            Y_aug = augmenter.augment(Y)

            try:
                f = interp1d(
                    df_species["time"].view(np.int64),
                    Y_aug,
                    kind="linear",
                    fill_value="extrapolate",
                )
                aug_data[col] = f(time_aug.view(np.int64))
            except Exception:
                aug_data[col] = np.full(max_len, np.nan)

        return pd.DataFrame(aug_data)

    def balance_species_ids(self, df, target_size=None, random_state=42):
        np.random.seed(random_state)

        species_counts = df.groupby("species")["id"].nunique()
        max_ids = target_size or species_counts.max()
        print(f"Target number of IDs per species: {max_ids}")

        balanced_mapping = {}

        for species, ids in df.groupby("species")["id"].unique().items():
            ids = np.array(ids)
            n_current = len(ids)
            if n_current < max_ids:
                extra = np.random.choice(ids, size=max_ids - n_current, replace=True)
                ids_bal = np.concatenate([ids, extra])
            else:
                ids_bal = ids[:max_ids]
            balanced_mapping[species] = ids_bal

        return balanced_mapping

    def augment(
        self,
        df: pd.DataFrame,
        spectral_bands=spectral_bands,
        max_len=152,
        random_state=42,
    ):
        np.random.seed(random_state)
        df["time"] = pd.to_datetime(df["time"])

        balanced_ids = self.balance_species_ids(df, random_state=random_state)
        augmented_dfs = []
        start_time = df.time.min()

        for species, ids_to_process in tqdm.tqdm(
            balanced_ids.items(), desc="Augmenting species"
        ):
            species_df = df[df["species"] == species]
            augmenter = self._make_augmenter()
            id_augmentation_counter = {}

            for tree_id in ids_to_process:
                df_species = species_df[species_df["id"] == tree_id]
                df_aug = self.resample_time_series(
                    df_species, spectral_bands, max_len, augmenter, start_time
                )
                is_duplicated = list(ids_to_process).count(tree_id) > 1

                if is_duplicated:
                    count = id_augmentation_counter.get(tree_id, 0) + 1
                    id_augmentation_counter[tree_id] = count
                    df_aug["id"] = f"{tree_id}_aug_{count}"
                else:
                    pass

                augmented_dfs.append(df_aug)

        df_final = pd.concat(augmented_dfs, ignore_index=True)
        return df_final

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run augmentation only for IDs with fewer samples than threshold."""
        if not self.on:
            return df

        id_counts = df.groupby("id").size()
        ids_to_augment = id_counts[id_counts < self.threshold].index
        ids_no_augment = id_counts[id_counts >= self.threshold].index

        df_to_augment = df[df["id"].isin(ids_to_augment)]
        df_no_augment = df[df["id"].isin(ids_no_augment)]

        df_augmented = self.augment(df_to_augment)
        df_final = pd.concat([df_no_augment, df_augmented], ignore_index=True)

        df_final["id"] = df_final["id"].astype(str)
        return df_final
