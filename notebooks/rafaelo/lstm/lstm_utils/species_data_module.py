from lstm_utils.species_dataset import SpeciesDataset
from lstm_utils.data_loader import DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import torch

pl.seed_everything(42)


# class SpeciesDataModule(pl.LightningDataModule):
#     def __init__(self, train_sequences, test_sequences, batch_size):
#         super().__init__()
#         self.train_sequences = train_sequences
#         self.test_sequences = test_sequences
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         self.train_dataset = SpeciesDataset(self.train_sequences)
#         self.test_dataset = SpeciesDataset(self.test_sequences)


# def train_dataloader(self):
#     return DataLoader(
#         self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
#     )

# def val_dataloader(self):
#     return DataLoader(
#         self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
#     )

# def test_dataloader(self):
#     return DataLoader(
#         self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
#     )


class SpeciesDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = SpeciesDataset(self.train_sequences)
        self.test_dataset = SpeciesDataset(self.test_sequences)

    def collate_fn(self, batch):
        sequences = [item["sequence"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch])
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        return {"sequence": padded_sequences, "label": labels}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
