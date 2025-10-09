from lstm_utils.species_dataset import SpeciesDataset
from lstm_utils.data_loader import DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import torch

pl.seed_everything(42)

class SpeciesDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, val_sequences, test_sequences, batch_size=64):
        super().__init__()
        self.train_sequences = train_sequences
        self.val_sequences = val_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SpeciesDataset(self.train_sequences)
        self.val_dataset = SpeciesDataset(self.val_sequences)
        self.test_dataset = SpeciesDataset(self.test_sequences)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=4, persistent_workers=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, persistent_workers=True
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=4, persistent_workers=True
        )