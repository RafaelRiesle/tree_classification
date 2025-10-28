import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from dtaidistance import dtw_ndim
from tqdm.auto import tqdm
from general_utils.constants import spectral_bands

class TimeSeriesAugmenter:
    def __init__(self, data: pd.DataFrame, 
                 id_col: str = "id", 
                 time_col: str = "time", 
                 value_cols: list = spectral_bands, 
                 class_col: str= "species"):
        self.id_col = id_col
        self.time_col = time_col
        self.value_cols = value_cols
        self.class_col = class_col
        self.data = data.copy()
        self.data[self.time_col] = pd.to_datetime(self.data[self.time_col])
        self.data = self.data.sort_values(by=[self.id_col, self.time_col])

    def _densify_series(self, series: pd.DataFrame, target_points: int, noise_level: float) -> pd.DataFrame:
        if series[self.time_col].duplicated().any():
            series = series.groupby(self.time_col)[self.value_cols].mean().reset_index()

        x_numeric = series[self.time_col].astype(np.int64) // 10**9
        new_x_numeric = np.linspace(x_numeric.min(), x_numeric.max(), target_points)
        new_time_index = pd.to_datetime(new_x_numeric, unit='s')
        
        densified_data = {self.time_col: new_time_index}

        for col in self.value_cols:
            y = series[col]
            if len(series) < 3:
                interpolated_values = np.interp(new_x_numeric, x_numeric, y)
            else:
                cs = CubicSpline(x_numeric, y)
                interpolated_values = cs(new_x_numeric)
            
            noise = np.random.normal(0, np.std(y) * noise_level, len(interpolated_values))
            densified_data[col] = interpolated_values + noise

        return pd.DataFrame(densified_data)

    def _create_synthetic_smote_series(self, base_series: np.ndarray, neighbor_series: np.ndarray) -> np.ndarray:
        lambda_val = np.random.rand()
        return base_series + lambda_val * (neighbor_series - base_series)


    def augment(self, target_points_per_series: int = 152, noise_level: float = 0.05, k_neighbors: int = 3):
        augmented_data_list = []
        grouped_data = self.data.groupby(self.id_col)
        for id_val, group in tqdm(grouped_data, desc="Processing Time Series"):
            densified_group = self._densify_series(group, target_points_per_series, noise_level)
            densified_group[self.id_col] = id_val
            densified_group[self.class_col] = group[self.class_col].iloc[0]
            augmented_data_list.append(densified_group)

        densified_df = pd.concat(augmented_data_list, ignore_index=True)

        id_to_class = self.data.drop_duplicates(subset=[self.id_col]).set_index(self.id_col)[self.class_col]
        class_counts = id_to_class.value_counts()
        

        target_count = class_counts.max()
        print(f"Class distribution before processing:\n{class_counts}")
        print(f"Target ID count: {target_count}\n")

        series_by_class = {}
        for class_val in id_to_class.unique():
            ids_in_class = id_to_class[id_to_class == class_val].index
            series_in_class = [
                densified_df[densified_df[self.id_col] == id_val][self.value_cols].values[:target_points_per_series]
                for id_val in ids_in_class
            ]
            series_by_class[class_val] = series_in_class

        newly_generated_series = []
        next_new_id = self.data[self.id_col].max() + 1 if pd.api.types.is_numeric_dtype(self.data[self.id_col]) else f"syn_{len(self.data[self.id_col].unique())}"

        for class_val, count in class_counts.items():
            if count < target_count:
                n_to_generate = target_count - count
                minority_series_list = series_by_class[class_val]
                pbar = tqdm(range(n_to_generate), desc=f"Generiere für Klasse '{class_val}'")
                
                if len(minority_series_list) < 2:
                    for _ in pbar:
                        base_series_data = minority_series_list[_ % len(minority_series_list)]
                        noise = np.random.normal(0, np.std(base_series_data, axis=0) * noise_level, base_series_data.shape)
                        new_df = pd.DataFrame(base_series_data + noise, columns=self.value_cols)
                        new_df[self.time_col] = pd.date_range(start=self.data[self.time_col].min(), periods=len(new_df), freq='H')
                        new_df[self.id_col] = next_new_id
                        new_df[self.class_col] = class_val
                        newly_generated_series.append(new_df)
                        next_new_id = (next_new_id + 1) if isinstance(next_new_id, int) else f"syn_{len(self.data[self.id_col].unique()) + _ + 1}"
                    continue

                for _ in pbar:
                    random_idx = np.random.randint(0, len(minority_series_list))
                    base_series = minority_series_list[random_idx]
                    distances = [dtw_ndim.distance(base_series, other) for other in minority_series_list]
                    neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
                    chosen_neighbor_idx = np.random.choice(neighbor_indices)
                    neighbor_series = minority_series_list[chosen_neighbor_idx]
                    synthetic_values = self._create_synthetic_smote_series(base_series, neighbor_series)
                    
                    new_df = pd.DataFrame(synthetic_values, columns=self.value_cols)
                    new_df[self.time_col] = pd.date_range(start=self.data[self.time_col].min(), periods=len(new_df), freq='H')
                    new_df[self.id_col] = next_new_id
                    new_df[self.class_col] = class_val
                    newly_generated_series.append(new_df)
                    next_new_id = (next_new_id + 1) if isinstance(next_new_id, int) else f"syn_{len(self.data[self.id_col].unique()) + _ + 1}"

        if newly_generated_series:
            final_df = pd.concat([densified_df] + newly_generated_series, ignore_index=True)
        else:
            final_df = densified_df

        print("\nAugmenting finished")
        final_id_to_class = final_df.drop_duplicates(subset=[self.id_col]).set_index(self.id_col)[self.class_col]
        print(f"Class distribution after processing:\n{final_id_to_class.value_counts()}")
        
        return final_df