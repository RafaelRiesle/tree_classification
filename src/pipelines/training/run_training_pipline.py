from pathlib import Path
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.run_evalution_for_best_model import (
    run_evaluation_for_best_model,
)
from models.lstm.pipeline.run_lstm_training import train_model, evaluate_model
from models.time_series.pipeline.run_pyts_model import PytsModelPipeline
from pipelines.processing.run_processing_pipeline import run_processing_pipeline
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)


class TrainingPipeline:
    def __init__(
        self,
        base_dir: Path = None,
        sample_size: int = None,
        remove_outliers: bool = False,
        force_split_creation: bool = False,
        force_preprocessing: bool = False,
        force_processing: bool = False,
        batch_size: int = 32,
        lr: float = 1e-3,
        max_epochs: int = 50,
    ):
        self.BASE_DIR = base_dir or Path(__file__).resolve().parents[3]
        self.DATA_PATH = self.BASE_DIR / "data/raw/raw_trainset.csv"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.RAW_DIR = self.BASE_DIR / "raw"
        self.SPLITS_PATH = self.BASE_DIR / "data/raw/splits"
        self.PREPROCESSED_PATH = self.BASE_DIR / "data/preprocessed"
        self.PROCESSED_DIR = self.DATA_DIR / "processed"

        self.paths = {
            "train_path": self.PROCESSED_DIR / "trainset.csv",
            "test_path": self.PROCESSED_DIR / "testset.csv",
            "val_path": self.PROCESSED_DIR / "valset.csv",
        }

        self.sample_size = sample_size
        self.remove_outliers = remove_outliers
        self.force_split_creation = force_split_creation
        self.force_preprocessing = force_preprocessing
        self.force_processing = force_processing
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs

    def run_preprocessing(self):
        run_preprocessing_pipeline(
            sample_size=self.sample_size,
            force_preprocessing=self.force_preprocessing,
            remove_outliers=self.remove_outliers,
            force_split_creation=self.force_split_creation,
        )

    def run_processing(self):
        run_processing_pipeline(force_processing=self.force_preprocessing)

    def run_ensemble_model(self):
        print("Training ensemble models...")
        run_ensemble(**self.paths)
        run_evaluation_for_best_model()
        print("Ensemble training complete.\n")

    def run_lstm(self):
        print("Training LSTM model...")
        model, data_module, data_info = train_model(
            **self.paths,
            batch_size=self.batch_size,
            lr=self.lr,
            max_epochs=self.max_epochs,
        )
        evaluate_model(model, data_module, data_info)
        print("LSTM training & evaluation complete.\n")

    def run_pyts(self):
        print("Training Pyts RandomForest model...")
        pyts_pipeline = PytsModelPipeline(
            train_path=self.paths["train_path"],
            test_path=self.paths["test_path"],
            valid_path=self.paths["val_path"],
        )
        pyts_pipeline.run_pipeline()
        print("Pyts RandomForest training complete.\n")

    def run(self):
        print("=== Starting Training Pipeline ===")
        self.run_preprocessing()
        self.run_processing()
        # self.run_pyts()
        # self.run_lstm()
        self.run_ensemble_model()
        print("=== Training Pipeline Finished ===")


if __name__ == "__main__":
    pipeline = TrainingPipeline(
        sample_size=None,
        remove_outliers=False,
        force_split_creation=True,
        force_preprocessing=True,
        force_processing=True,
    )
    pipeline.run()
