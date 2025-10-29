import pandas as pd
import numpy as np
from tsaug import Drift, AddNoise
from scipy.interpolate import interp1d
import warnings
from tqdm import tqdm
from collections import Counter
from general_utils.constants import spectral_bands

warnings.filterwarnings("ignore")


class DataAugmentation:
    def __init__(self, on=True, scale=0.02, drift=0.01, random_state=42):
        self.on = on
        self.scale = scale
        self.drift = drift
        self.rng = np.random.default_rng(
            random_state
        )  # Besserer Weg zur Verwaltung von Zufälligkeit

    def _make_augmenter(self):
        """Erstellt einen Augmenter mit Drift und Rauschen."""
        # WICHTIG: Wir erzeugen hier für jeden Aufruf neue Integer-Seeds aus unserem Generator.
        # So ist jede Augmentierung anders, aber der gesamte Lauf bleibt reproduzierbar.
        seed1 = self.rng.integers(np.iinfo(np.int32).max)
        seed2 = self.rng.integers(np.iinfo(np.int32).max)

        return Drift(max_drift=self.drift, seed=seed1) + AddNoise(
            scale=self.scale, seed=seed2
        )

    def _resample_time_series(
        self, df_id: pd.DataFrame, time_index_aug: pd.DatetimeIndex, augmenter
    ) -> pd.DataFrame:
        """Resampelt und augmentiert die Zeitreihe einer einzelnen ID."""
        df_id = df_id.sort_values("time").reset_index(drop=True)

        # Sicherheitsabfrage: Mindestens 2 Punkte für Interpolation nötig
        if len(df_id) < 2:
            return None

        aug_data = {"time": time_index_aug}

        for col in spectral_bands:
            # Originaldaten vorbereiten
            y = df_id[col].interpolate(method="linear").to_numpy(dtype=np.float64)
            x = df_id["time"].view(np.int64)

            # Augmentierung anwenden
            y_aug = augmenter.augment(y.reshape(1, -1, 1)).flatten()

            # Interpolation - SICHERER ANSATZ OHNE EXTRAPOLATION
            f = interp1d(
                x,
                y_aug,
                kind="linear",
                bounds_error=False,  # Kein Fehler bei Werten außerhalb des Bereichs
                fill_value=(y_aug[0], y_aug[-1]),  # Fülle mit erstem/letztem Wert
            )
            aug_data[col] = f(time_index_aug.view(np.int64))

        return pd.DataFrame(aug_data)

    def run(
        self, df: pd.DataFrame, target_len=152, target_ids_per_species=None
    ) -> pd.DataFrame:
        """
        Führt die Augmentierung durch, um die Anzahl der IDs pro Spezies auszugleichen.
        Alle Zeitreihen werden auf eine einheitliche Länge und Zeitachse gebracht.
        """
        if not self.on:
            return df

        df["time"] = pd.to_datetime(df["time"])

        # 1. Bestimme die Ziel-ID-Anzahl pro Spezies
        species_id_counts = df.groupby("species")["id"].nunique()
        if target_ids_per_species is None:
            target_ids_per_species = species_id_counts.max()
        print(f"Ziel-Anzahl von IDs pro Spezies: {target_ids_per_species}")

        # 2. Erstelle eine Liste der zu erzeugenden IDs
        ids_to_generate = []
        original_ids = []
        for species, group in df.groupby("species"):
            unique_ids = group["id"].unique()
            original_ids.extend(unique_ids)

            n_current = len(unique_ids)
            n_to_add = target_ids_per_species - n_current

            if n_to_add > 0:
                # Füge Original-IDs hinzu
                ids_to_generate.extend([(species, old_id) for old_id in unique_ids])
                # Füge neue, zu generierende IDs hinzu (basierend auf zufälliger Auswahl)
                chosen_ids = self.rng.choice(unique_ids, size=n_to_add, replace=True)
                ids_to_generate.extend([(species, old_id) for old_id in chosen_ids])
            else:
                # Wenn Spezies bereits genug IDs hat, nimm alle Original-IDs
                # Optional: Man könnte hier auch auf target_ids_per_species undersamplen
                ids_to_generate.extend([(species, old_id) for old_id in unique_ids])

        # 3. Bereite die finale Zeitachse vor (global für alle!)
        start_time = df["time"].min()
        time_index_aug = pd.date_range(start=start_time, periods=target_len, freq="14D")

        # 4. Verarbeite alle IDs (Originale und zu erzeugende)
        augmented_dfs = []
        id_augmentation_counter = Counter()

        # Gruppiere nach der Original-ID für effizienten Zugriff
        df_grouped_by_id = df.groupby("id")

        print("Resampling und Augmentierung aller Zeitreihen...")
        for species, tree_id in tqdm(ids_to_generate):
            df_id_original = df_grouped_by_id.get_group(tree_id)

            # Jeder Durchlauf bekommt einen neuen Augmenter für unterschiedliche Ergebnisse
            augmenter = self._make_augmenter()

            df_aug = self._resample_time_series(
                df_id_original, time_index_aug, augmenter
            )

            if df_aug is None:
                continue  # Überspringe, wenn die Zeitreihe zu kurz war

            # Zähle, wie oft diese ID schon verwendet wurde, um neue, einzigartige IDs zu erstellen
            id_augmentation_counter[tree_id] += 1
            count = id_augmentation_counter[tree_id]

            if count > 1:
                # Dies ist eine augmentierte Kopie
                df_aug["id"] = f"{tree_id}_aug_{count - 1}"
            else:
                # Dies ist die (resampelte) Original-ID
                df_aug["id"] = tree_id

            df_aug["species"] = species
            augmented_dfs.append(df_aug)

        df_final = pd.concat(augmented_dfs, ignore_index=True)
        return df_final
