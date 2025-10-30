import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.processing_steps.interpolation import Interpolation
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.features.temporal_features import TemporalFeatures
from models.baseline_model.calculate_keyfigures import StatisticalFeatures
from models.baseline_model.baseline_model_utils import drop_unwanted_columns
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")


def train_and_score_fast(X_train, y_train, X_test, y_test, num_classes):
    train_cols = X_train.columns
    X_test_aligned = X_test.reindex(columns=train_cols).fillna(0)

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=10,
        random_state=42,
        eval_metric="mlogloss",
        objective="multi:softprob",
        num_class=num_classes,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test_aligned)
    return accuracy_score(y_test, preds)


def get_interval_months(start, length):
    return [(start + i - 1) % 12 + 1 for i in range(length)]


class Config:
    ID_COLUMN = "id"
    TARGET_COLUMN = "species"
    TEMPORAL_UNIT_COLUMN = "month"
    TEST_RATIO = 0.2
    RANDOM_SEED = 42
    TEMP_DATA_DIR = Path("./temp_data_for_pipeline")


def get_processing_steps() -> List[Any]:
    return [
        BasicFeatures(on=True),
        Interpolation(on=True),
        CalculateIndices(on=True),
        TemporalFeatures(on=True),
    ]


def split_data(df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_unique_ids = df[config.ID_COLUMN].unique()
    rng = np.random.default_rng(config.RANDOM_SEED)
    rng.shuffle(all_unique_ids)

    num_test = max(1, int(config.TEST_RATIO * len(all_unique_ids)))
    global_test_ids = all_unique_ids[:num_test]

    global_test_df = df[df[config.ID_COLUMN].isin(global_test_ids)].copy()
    df_train_pool = df[~df[config.ID_COLUMN].isin(global_test_ids)].copy()

    return df_train_pool, global_test_df


def process_and_aggregate(
    df: pd.DataFrame, config: Config, steps: List[Any], bands_and_indices: List[str]
) -> pd.DataFrame:
    config.TEMP_DATA_DIR.mkdir(exist_ok=True)
    temp_path = config.TEMP_DATA_DIR / "temp_processing_file.csv"
    df.to_csv(temp_path, index=False)

    pipeline = ProcessingPipeline(path=temp_path, steps=steps)
    processed_df = pipeline.run()
    processed_df = drop_unwanted_columns(processed_df)

    sf = StatisticalFeatures()
    aggregated_df = sf.calculate_keyfigures_per_id(processed_df, bands_and_indices)

    temp_path.unlink()
    return aggregated_df.copy()


def encode_labels(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    encoded_col_name = f"{config.TARGET_COLUMN}_encoded"

    train_df[encoded_col_name] = le.fit_transform(train_df[config.TARGET_COLUMN])
    test_df[encoded_col_name] = le.transform(test_df[config.TARGET_COLUMN])

    return train_df, test_df, le


def find_best_temporal_intervals(
    train_agg: pd.DataFrame,
    train_pool_raw: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_classes: int,
    config: Config,
) -> Tuple[Dict[float, float], Dict[float, List[Any]]]:
    results = {}
    intervals = {}

    all_temporal_units = sorted(train_pool_raw[config.TEMPORAL_UNIT_COLUMN].unique())
    num_units = len(all_temporal_units)

    total_steps = num_units * num_units

    with tqdm(total=total_steps, desc="Training", ncols=100) as pbar:
        for length in range(1, num_units + 1):
            best_acc = -1
            best_interval_units = None

            for start_idx in range(num_units):
                interval_units = [
                    all_temporal_units[(start_idx + i) % num_units]
                    for i in range(length)
                ]

                ids_in_interval = train_pool_raw[
                    train_pool_raw[config.TEMPORAL_UNIT_COLUMN].isin(interval_units)
                ][config.ID_COLUMN].unique()
                train_subset = train_agg[
                    train_agg[config.ID_COLUMN].isin(ids_in_interval)
                ]

                if train_subset.empty:
                    acc = 0.0
                else:
                    X_train_subset = train_subset.drop(
                        columns=[
                            config.TARGET_COLUMN,
                            f"{config.TARGET_COLUMN}_encoded",
                            config.ID_COLUMN,
                        ]
                    )
                    y_train_subset = train_subset[f"{config.TARGET_COLUMN}_encoded"]
                    acc = train_and_score_fast(
                        X_train_subset, y_train_subset, X_test, y_test, num_classes
                    )

                if acc > best_acc:
                    best_acc = acc
                    best_interval_units = interval_units

                pbar.update(1)

            percent = length / num_units * 100
            results[round(percent, 1)] = best_acc
            intervals[round(percent, 1)] = best_interval_units

    return results, intervals
