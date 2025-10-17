import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from models.lstm.lstm_utils.data_module import SpeciesDataModule
from models.lstm.lstm_utils.species_predictor import SpeciesPredictor


class LSTMTrainer:
    def __init__(
        self,
        train_sequences,
        val_sequences,
        test_sequences,
        n_features,
        n_classes,
        class_weights,
        batch_size=16,
        lr=1e-3,
        max_epochs=5,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_weights = class_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data module
        self.data_module = SpeciesDataModule(
            train_sequences, val_sequences, test_sequences, batch_size
        )
        self.model = SpeciesPredictor(n_features, n_classes, lr, class_weights)
        self.model.to(self.device)

    def train(self):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="trained_model/",
            filename="species_model-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            log_every_n_steps=1,
            deterministic=True,
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval="epoch"),
                EarlyStopping(monitor="val_acc", patience=10),
            ],
        )

        trainer.fit(self.model, datamodule=self.data_module)
        trainer.test(self.model, datamodule=self.data_module)

    def get_model(self):
        return self.model, self.data_module
