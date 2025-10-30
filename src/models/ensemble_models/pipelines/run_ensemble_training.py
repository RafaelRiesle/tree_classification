from pathlib import Path
from models.ensemble_models.experiments.run_ensemble import run_ensemble
from models.ensemble_models.evaluation.run_evalution_for_best_model import (
    run_evaluation_for_best_model,
)


class TrainingPipeline:
    def __init__(
        self,
        base_dir: Path = None,
    ):
        # === Paths ===
        self.base_dir = base_dir or Path(__file__).resolve().parents[4]
        self.data_dir = self.base_dir / "data"

        self.raw_dir = self.data_dir / "raw"
        self.preprocessed_dir = self.data_dir / "preprocessed"
        self.processed_dir = self.data_dir / "processed"

        self.paths = {
            "train_path": self.processed_dir / "trainset.csv",
            "test_path": self.processed_dir / "testset.csv",
            "val_path": self.processed_dir / "valset.csv",
        }

    def run_ensemble_models(self):
        print("[3] Training ensemble models...")
        run_ensemble(**self.paths)
        # run_evaluation_for_best_model()
        print("[3] Ensemble training complete.\n")


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_ensemble_models()
