from pathlib import Path
from pipelines.processing.processing_pipeline import ProcessingPipeline
from pipelines.processing.features.basic_features import BasicFeatures
from pipelines.processing.processing_steps.aggregation import TimeSeriesAggregate
from pipelines.processing.processing_steps.interpolate_nans import InterpolateNaNs
from pipelines.processing.processing_steps.smoothing import Smooth
from pipelines.processing.features.spectral_indices import CalculateIndices
from pipelines.processing.features.temporal_features import TemporalFeatures
from pipelines.processing.processing_steps.interpolation import Interpolation
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
new_data_path = BASE_DIR / "data/val/FINAL_Validierungs_Datensatz.csv"
processed_output_path = BASE_DIR / "data/val/val_processed.csv"

test_steps = [
    BasicFeatures(on=True),
    TimeSeriesAggregate(on=True, freq=2, method="mean"),
    InterpolateNaNs(on=True, method="linear"),
    Smooth(on=True, overwrite=True),
    CalculateIndices(on=True),
    TemporalFeatures(on=True),
    Interpolation(on=True),
]

pipeline = ProcessingPipeline(path=new_data_path, steps=test_steps)
df_processed = pipeline.run()
processed_output_path.parent.mkdir(parents=True, exist_ok=True)
df_processed.to_csv(processed_output_path, index=False)
print(
    f"✓ Finished processing unseen data; saved to {processed_output_path}, shape={df_processed.shape}"
)
