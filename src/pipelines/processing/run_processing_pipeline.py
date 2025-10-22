from pathlib import Path
from pipelines.preprocessing.run_preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
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
processed_dir = BASE_DIR / "data/processed"

paths = {
            "train_path": processed_dir / "trainset.csv",
            "test_path": processed_dir / "testset.csv",
            "val_path": processed_dir / "valset.csv",
        }


def run_preprocessing():
    print("[1] Running preprocessing...")
    run_preprocessing_pipeline(
        data_path=RAW_DIR / "raw_trainset.csv",
        splits_output_path=RAW_DIR / "splits",
        preprocessed_output_path=PREPROCESSED_DIR,
        sample_size=None,
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
            InterpolateNaNs(on=True, method="quadratic"),   
            CalculateIndices(on=True),
            TemporalFeatures(on=True),
            Interpolation(on=True),
        ]


        train_steps = [
            TimeSeriesFilter(on=True),   
            BasicFeatures(on=True),
            OldDisturbancePruner(on=True),
            CalculateIndices(on=True),
            DetectDisturbedTrees(on=True),
            AdjustLabels(on=True),
            DataAugmentation(on=True, threshold=150),
            TimeSeriesAggregate(on=True, freq=2, method="mean"),
            InterpolateNaNs(on=True, method="quadratic"),
            Smooth(on=True, overwrite=True),
            Interpolation(on=True),
            CalculateIndices(on=True), # Second time because of augmentation
            TemporalFeatures(on=True),  
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
        print(f"✓ Saved processed {split_name} set to: {output_path}")


def run_training_pipeline():
    print("=== Starting Processing Pipeline ===")
    run_preprocessing()
    run_processing()
    print("=== Processing Finished ===")


if __name__ == "__main__":
    run_training_pipeline()
