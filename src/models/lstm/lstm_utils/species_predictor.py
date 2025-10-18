from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from models.lstm.lstm_utils.lstm_model import ImprovedSequenceModel
import numpy as np


class SpeciesPredictor(pl.LightningModule):
    def __init__(self, n_features, n_classes, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = ImprovedSequenceModel(n_features, n_classes)
        self.lr = lr

        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)

        self.train_acc_history, self.val_acc_history = [], []
        self.test_acc_history, self.train_loss_history = [], []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        outputs = self.forward(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.train_acc(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        if not hasattr(self, "_train_loss_accum"):
            self._train_loss_accum = []
        self._train_loss_accum.append(loss.detach().cpu().item())
        return loss

    def on_train_epoch_end(self):
        self.train_loss_history.append(np.mean(self._train_loss_accum))
        self.train_acc_history.append(self.train_acc.compute().item())
        self.train_acc.reset()
        self._train_loss_accum = []

    def validation_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        outputs = self.forward(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.val_acc_history.append(self.val_acc.compute().item())
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        sequences, labels = batch["sequence"], batch["label"]
        outputs = self.forward(sequences)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        acc = self.test_acc(preds, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        self.test_acc_history.append(self.test_acc.compute().item())
        self.test_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }
