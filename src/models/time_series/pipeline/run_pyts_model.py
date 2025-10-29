import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from pathlib import Path
from models.time_series.ts_randomforrest_utils.dataset_builder import PytsDatasetBuilder
from models.time_series.ts_randomforrest_utils.model import FlattenedRandomForestModel


BASE_DIR = Path(__file__).parents[4]
TRAIN_PATH = BASE_DIR / "data/processed/trainset.csv"
TEST_PATH = BASE_DIR / "data/processed/testset.csv"
VAL_PATH = BASE_DIR / "data/processed/valset.csv"
SAVE_DIR = BASE_DIR / "data/pyts/models"


class PytsModelPipeline:
    def __init__(
        self,
        train_path: str,
        test_path: str,
        valid_path: str,
        model_save_dir: str = SAVE_DIR,
        random_state: int = 42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.model_save_dir = model_save_dir
        self.random_state = random_state

        self.dataset_builder = None
        self.datasets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.model = None
        self.label_encoder = None
        self.best_params = None
        self.best_score = None
        self.save_path = None

    def prepare_data(self):
        print("Loading and preparing datasets...")
        self.dataset_builder = PytsDatasetBuilder(
            self.train_path, self.test_path, self.valid_path
        )
        self.datasets = self.dataset_builder.build()
        self.label_encoder = self.dataset_builder.preprocessor.label_encoder

    def train(
        self, param_grid: Dict[str, List[Any]], cv: int = 3, scoring: str = "accuracy"
    ):
        self.model = FlattenedRandomForestModel(random_state=self.random_state)
        X_train, y_train = self.datasets["train"]
        self.best_params, self.best_score = self.model.grid_search(
            X_train, y_train, param_grid=param_grid, cv=cv, scoring=scoring
        )

    def evaluate(self):
        for set_name in ["test", "valid"]:
            X, y = self.datasets[set_name]
            self.model.predict(X)
            self.model.evaluate(
                X, y, label_encoder=self.label_encoder, set_name=set_name.capitalize()
            )

        self.model.save_feature_importances_plot(
            feature_names=self.dataset_builder.preprocessor.feature_cols,
            n_timestamps=self.datasets["train"][0].shape[1],
            folder=self.model_save_dir / "feature_importances_plots",
        )

    def save_model(self):
        self.save_path = self.model.save(
            folder=self.model_save_dir, label_encoder=self.label_encoder
        )

    def load_and_test(self):
        loaded_model = FlattenedRandomForestModel()
        loaded_label_encoder = loaded_model.load(self.save_path)
        X_test, y_test = self.datasets["test"]
        loaded_model.evaluate(
            X_test, y_test, label_encoder=loaded_label_encoder, set_name="Test (Loaded)"
        )

    def define_grid_search_params(self) -> Dict[str, List[Any]]:
        return {"n_estimators": [300], "criterion": ["gini"]}

    def run_pipeline(self):
        self.prepare_data()
        self.train(param_grid=self.define_grid_search_params())
        self.evaluate()
        self.save_model()
        self.load_and_test()


if __name__ == "__main__":
    pipeline = PytsModelPipeline(
        train_path=TRAIN_PATH, test_path=TEST_PATH, valid_path=VAL_PATH
    )
    pipeline.run_pipeline()
