import numpy as np
from typing import List, Tuple, Dict
from models.time_series.ts_randomforrest_utils.time_series_preprocessor import TimeSeriesPreprocessor
class PytsDatasetBuilder:
    """Manages train/test/validation dataset preparation."""

    def __init__(
        self,
        train_path: str,
        test_path: str,
        valid_path: str,
        feature_cols: List[str] = None,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        self.preprocessor = TimeSeriesPreprocessor(feature_cols)

    def build(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Creates X_train, y_train, X_test, y_test, X_valid, y_valid."""
        print("Preparing training data (fit)...")
        X_train, y_train = self.preprocessor.prepare_dataset(self.train_path, fit=True)

        print("Preparing test data (transform only)...")
        X_test, y_test = self.preprocessor.prepare_dataset(self.test_path, fit=False)

        print("Preparing validation data (transform only)...")
        X_valid, y_valid = self.preprocessor.prepare_dataset(self.valid_path, fit=False)

        return {
            "train": (X_train, y_train),
            "test": (X_test, y_test),
            "valid": (X_valid, y_valid),
        }
