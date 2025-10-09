# from lstm_utils.species_dataset import PaddedSpeciesDataset
from lstm_utils.data_loader import DataLoader
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch
import pandas as pd 
import numpy as np

# pl.seed_everything(42)

# class SpeciesDataModule(pl.LightningDataModule):
#     def __init__(self, train_sequences, val_sequences, test_sequences, batch_size=64):
#         super().__init__()
#         self.train_sequences = train_sequences
#         self.val_sequences = val_sequences
#         self.test_sequences = test_sequences
#         self.batch_size = batch_size

#     def setup(self, stage=None):
#         self.train_dataset = SpeciesDataset(self.train_sequences)
#         self.val_dataset = SpeciesDataset(self.val_sequences)
#         self.test_dataset = SpeciesDataset(self.test_sequences)

#     def train_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.train_dataset, batch_size=self.batch_size, shuffle=True,
#             num_workers=4, persistent_workers=True
#         )

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.val_dataset, batch_size=self.batch_size, shuffle=False,
#             num_workers=4, persistent_workers=True
#         )

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.test_dataset, batch_size=self.batch_size, shuffle=False,
#             num_workers=4, persistent_workers=True
#         )


class PaddedSpeciesDataset(Dataset):
    def __init__(self, sequences, pad_value=0.0, max_len=None):
        self.sequences = sequences
        self.pad_value = pad_value
        self.max_len = max_len if max_len is not None else max(len(X) for X, _ in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        X, y = self.sequences[idx]
        if isinstance(X, pd.DataFrame):
            X = X.values
        seq_len, n_features = X.shape
        padded_X = np.full((self.max_len, n_features), self.pad_value, dtype=np.float32)
        padded_X[:seq_len, :] = X
        return {
            "sequence": torch.tensor(padded_X, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.long),
            "length": seq_len,
        }


class SpeciesDataModule(pl.LightningDataModule):
    def __init__(self, train_seqs, val_seqs, test_seqs, batch_size=32):
        super().__init__()
        self.train_seqs = train_seqs
        self.val_seqs = val_seqs
        self.test_seqs = test_seqs
        self.batch_size = batch_size

        # max_len berechnen
        self.max_len = max(len(X) for X, _ in train_seqs)

    def setup(self, stage=None):
        self.train_dataset = PaddedSpeciesDataset(self.train_seqs, max_len=self.max_len)
        self.val_dataset = PaddedSpeciesDataset(self.val_seqs, max_len=self.max_len)
        self.test_dataset = PaddedSpeciesDataset(self.test_seqs, max_len=self.max_len)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
