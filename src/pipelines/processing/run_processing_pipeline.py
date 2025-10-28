from pathlib import Path
from pipelines.processing.processing_steps.interpolation import Interpolation
from pipelines.processing.data_reduction.detect_disturbed_trees import (
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
from pipelines.processing.processing_steps.smoothing import Smooth
from pipelines.processing.data_reduction.old_disturbance_pruner import (
    OldDisturbancePruner,
)
from pipelines.processing.data_reduction.timeseries_filter import TimeSeriesFilter

BASE_DIR = Path(__file__).resolve().parents[3]
RAW_DIR = BASE_DIR / "data/raw"
PREPROCESSED_DIR = BASE_DIR / "data/preprocessed"
PROCESSED_DIR = BASE_DIR / "data/processed"

paths = {
    "train_path": PROCESSED_DIR / "trainset.csv",
    "test_path": PROCESSED_DIR / "testset.csv",
    "val_path": PROCESSED_DIR / "valset.csv",
}


def run_processing():
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

    test_steps = [
        BasicFeatures(on=True),
        TimeSeriesAggregate(on=True, freq=2, method="mean"),
        InterpolateNaNs(on=True, method="linear"),
        Smooth(on=True, overwrite=True),
        CalculateIndices(on=True),
        TemporalFeatures(on=False),
        Interpolation(on=True),
    ]

    train_steps = [
        TimeSeriesFilter(on=True, max_median_diff_days=25),
        BasicFeatures(on=True),
        OldDisturbancePruner(on=False),
        CalculateIndices(on=True),
        DetectDisturbedTrees(on=False),
        AdjustLabels(on=False),
        DataAugmentation(on=False, threshold=150),
        TimeSeriesAggregate(on=True, freq=2, method="mean"),
        InterpolateNaNs(on=True, method="linear"),
        Smooth(on=True, overwrite=True),
        Interpolation(on=True),
        CalculateIndices(on=True),
        TemporalFeatures(on=False),
    ]
    for split_name, path_dict in split_to_paths.items():
        input_path = path_dict["input"]
        output_path = path_dict["output"]
        if split_name == "train":
            steps = train_steps
        else:
            steps = test_steps

        print(f"→ Processing {split_name} set: {input_path.name}")

        pipeline = ProcessingPipeline(path=input_path, steps=steps)
        df_processed = pipeline.run()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"✓ Saved processed {split_name}; Shape: {df_processed.shape}")


def run_processing_pipeline(force_processing=True):
    print("=== Starting Processing Pipeline ===")
    processed_files_exist = all(
        paths[key].exists() for key in ["train_path", "test_path", "val_path"]
    )

    if processed_files_exist and not force_processing:
        print("[2] Skipping processing — existing processed files found.")
        return
    run_processing()
    print("=== Processing Finished ===")


if __name__ == "__main__":
    run_processing_pipeline()
