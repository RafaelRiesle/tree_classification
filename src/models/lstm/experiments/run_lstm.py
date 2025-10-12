# train.py
import pytorch_lightning as pl
from models.lstm.lstm_utils.species_data_module import SpeciesDataModule
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor
from models.lstm.pipeline.pipeline import prepare_data




if __name__ == "__main__":
    data = prepare_data(
        "../../../data/baseline_training/trainset.csv",
        "../../../data/baseline_training/valset.csv",
        "../../../data/baseline_training/testset.csv",
    )

    n_features = len(data["feature_columns"])
    batch_size = 32
    lr = 1e-3
    max_epochs = 2

    data_module = SpeciesDataModule(
        data["train_sequences"],
        data["val_sequences"],
        data["test_sequences"],
        batch_size=batch_size,
    )

    model = SpeciesPredictor(n_features=n_features, n_classes=data["n_classes"], lr=lr)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if data["device"] == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=10,
        deterministic=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint("trained_model/species_model.ckpt")
