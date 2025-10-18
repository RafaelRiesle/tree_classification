from pathlib import Path
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
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
from pipelines.processing.processing_steps.aggregation import TimeSeriesAggregate
from pipelines.processing.processing_steps.interpolate_nans import InterpolateNaNs

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data/raw"
PREPROCESSED_DIR = BASE_DIR / "data/preprocessed"
PROCESSED_DIR = BASE_DIR / "data/processed"

paths = {split: PROCESSED_DIR / f"{split}set.csv" for split in ["train", "test", "val"]}


def run_preprocessing():
    print("[1] Running preprocessing...")
    run_preprocessing_pipeline(
        data_path=RAW_DIR / "raw_trainset.csv",
        splits_output_path=RAW_DIR / "splits",
        preprocessed_output_path=PREPROCESSED_DIR,
        sample_size=300,
        train_ratio=0.7,
        test_ratio=0.2,
        val_ratio=0.1,
        remove_outliers=False,
        contamination=0.05,
        force_split_creation=True,
    )
    print("[1] Preprocessing complete.\n")

def run_processing():
    print("[2] Running processing for train, test, val sets...")

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

    for split in ["train", "test", "val"]:
        input_path = PREPROCESSED_DIR / f"{split}set.csv"
        output_path = paths[split]

        print(f"→ Processing {split} set: {input_path.name}")
        pipeline = ProcessingPipeline(path=input_path, steps=steps)
        df_processed = pipeline.run()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"✓ Saved {split} set to: {output_path}")

    print("[2] Processing complete.\n")


def run_training_pipeline():
    print("=== Starting Processing Pipeline ===")
    run_preprocessing()
    run_processing()
    print("=== Processing Finished ===")


if __name__ == "__main__":
    run_training_pipeline()
