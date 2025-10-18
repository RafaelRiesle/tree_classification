from pathlib import Path
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.run_evalution_for_best_model import (
    run_evaluation_for_best_model,
)
from models.lstm.pipeline.run_lstm_training import train_model, evaluate_model
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.processing_steps.aggregation import TimeSeriesAggregate
from pipelines.processing.processing_steps.interpolate_nans import InterpolateNaNs
from pipelines.processing.processing_steps.interpolation import Interpolation
from pipelines.processing.processing_steps.data_augmentation import DataAugmentation
from pipelines.processing.processing_steps.detect_disturbed_trees import (
    DetectDisturbedTrees,
)
from pipelines.processing.processing_steps.adjust_labels import AdjustLabels


class TrainingPipeline:
    def __init__(
        self,
        base_dir: Path = None,
        sample_size: int = 100,
        train_ratio: float = 0.7,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        remove_outliers: bool = False,
        force_split_creation: bool = True,
        batch_size: int = 64,
        lr: float = 1e-3,
        max_epochs: int = 2,
    ):
        # === Paths ===
        self.base_dir = base_dir or Path(__file__).resolve().parents[3]
        self.data_dir = self.base_dir / "data"

        self.raw_dir = self.data_dir / "raw"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.processed_dir = self.data_dir / "processed"

        self.paths = {
            "train_path": self.processed_dir / "trainset.csv",
            "test_path": self.processed_dir / "testset.csv",
            "val_path": self.processed_dir / "valset.csv",
        }

        # === Parameters ===
        self.sample_size = sample_size
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.remove_outliers = remove_outliers
        self.force_split_creation = force_split_creation
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs

    def run_preprocessing(self):
        print("[1] Running preprocessing...")

        run_preprocessing_pipeline(
            data_path=self.raw_dir / "raw_trainset.csv",
            splits_output_path=self.raw_dir / "splits",
            preprocessed_output_path=self.preprocessed_dir,
            sample_size=self.sample_size,
            train_ratio=self.train_ratio,
            test_ratio=self.test_ratio,
            val_ratio=self.val_ratio,
            remove_outliers=self.remove_outliers,
            contamination=0.05,
            force_split_creation=self.force_split_creation,
        )
        print("[1] Preprocessing complete.\n")

    def run_processing(self):
        print("[2] Running processing for train, test and val datasets...")

        split_to_paths = {
            "train": {
                "input": self.preprocessed_dir / "trainset.csv",
                "output": self.paths["train_path"],
            },
            "test": {
                "input": self.preprocessed_dir / "testset.csv",
                "output": self.paths["test_path"],
            },
            "val": {
                "input": self.preprocessed_dir / "valset.csv",
                "output": self.paths["val_path"],
            },
        }

        steps = [
            BasicFeatures(on=False),
            TimeSeriesAggregate(on=True, freq=2, method="mean"),
            CalculateIndices(on=True),
            InterpolateNaNs(on=True, method="quadratic"),
            TemporalFeatures(on=True),
            DataAugmentation(on=False),
            Interpolation(on=True),
            DetectDisturbedTrees(on=False),
            AdjustLabels(on=False),
        ]

        for split_name, path_dict in split_to_paths.items():
            input_path = path_dict["input"]
            output_path = path_dict["output"]

            print(f"→ Processing {split_name} set: {input_path.name}")

            pipeline = ProcessingPipeline(path=input_path, steps=steps)
            df_processed = pipeline.run()

            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_processed.to_csv(output_path, index=False)
            print(f"✓ Saved processed {split_name} set to: {output_path}")

        print("[2] Processing complete.\n")

    def run_ensemble_models(self):
        print("[3] Training ensemble models...")
        run_ensemble(**self.paths)
        run_evaluation_for_best_model()
        print("[3] Ensemble training complete.\n")

    def run_lstm_models(self):
        print("[4] Training LSTM model...")
        model, data_module, data_info = train_model(
            **self.paths,
            batch_size=self.batch_size,
            lr=self.lr,
            max_epochs=self.max_epochs,
        )
        evaluate_model(model, data_module, data_info)
        print("[4] LSTM training & evaluation complete.\n")

    def run(self):
        print("=== Starting Training Pipeline ===")
        self.run_preprocessing()
        self.run_processing()
        self.run_ensemble_models()
        self.run_lstm_models()
        print("=== Training Pipeline Finished ===")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
