import pandas as pd
from pathlib import Path

# === IMPORTS FÜR PIPELINES & MODELLE ===
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.evaluate_models import run_ensemble_evaluation
from models.lstm.experiments.run_lstm import run_lstm
from models.lstm.validation.evaluate import run_lstm_evaluation

from pipelines.processing.processing_steps.interpolation import Interpolation
from pipelines.processing.processing_steps.detect_disturbed_trees import (
    DetectDisturbedTrees,
)
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.processing_steps.data_augmentation import DataAugmentation
from pipelines.processing.processing_steps.adjust_labels import AdjustLabels
from pipelines.processing.processing_pipeline import ProcessingPipeline



BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
PROCESSED_DIR = DATA_DIR / "processed"

paths = {
    "train_path": PROCESSED_DIR / "trainset.csv",
    "test_path": PROCESSED_DIR / "testset.csv",
    "val_path": PROCESSED_DIR / "valset.csv",
}


def run_preprocessing():
    print("[1] Running preprocessing...")
    run_preprocessing_pipeline(
        data_path=RAW_DIR / "raw_trainset.csv",
        splits_output_path=RAW_DIR / "splits",
        preprocessed_output_path=PREPROCESSED_DIR,
        sample_size=100,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        remove_outliers=False,
        contamination=0.05,
        force_split_creation=False,
    )
    print("[1] Preprocessing complete.\n")

def run_processing():
    print("[2] Running processing for train, test and val datasets...")

    steps = [
        BasicFeatures(on=True),
        TemporalFeatures(on=True),
        Interpolation(on=True),
        DataAugmentation(on=True),
        CalculateIndices(on=True),
        # AdjustLabels(on=False),
        # DetectDisturbedTrees(on=False),
    ]

    split_to_paths = {
        "train": {
            "input": PREPROCESSED_DIR / "trainset.csv",
            "output": paths["train_path"],
        },
        "test": {
            "input": PREPROCESSED_DIR / "testset.csv",
            "output": paths["test_path"],
        },
        "val": {
            "input": PREPROCESSED_DIR / "valset.csv",
            "output": paths["val_path"],
        },
    }

    for split_name, path_dict in split_to_paths.items():
        input_path = path_dict["input"]
        output_path = path_dict["output"]

        print(f"→ Processing {split_name} set: {input_path.name}")

        pipeline = ProcessingPipeline(path=input_path, steps=steps)
        df_processed = pipeline.run()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"✓ Saved processed {split_name} set to: {output_path}")

    print("[2] Processing for all datasets complete.\n")


def run_ensemble_models():
    print("[3] Training ensemble models...")
    run_ensemble(**paths)
    run_ensemble_evaluation()
    print("[3] Ensemble training complete.\n")



def run_lstm_models():
    print("[4] Training LSTM model...")
    run_lstm(**paths, batch_size=16, lr=1e-3, max_epochs=2)
    run_lstm_evaluation(**paths, batch_size=50)
    print("[4] LSTM training complete.\n")


def run_training_pipeline():
    print("=== Starting Training Pipeline ===")
    run_preprocessing()
    run_processing()
    run_ensemble_models()
    run_lstm_models()
    print("=== Training Pipeline Finished ===")

if __name__ == "__main__":
    run_training_pipeline()
