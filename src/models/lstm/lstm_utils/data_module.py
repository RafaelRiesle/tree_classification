from models.lstm.lstm_utils.padded_dataset import PaddedSpeciesDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

class SpeciesDataModule(pl.LightningDataModule):
    def __init__(self, train_seqs, test_seqs, val_seqs, batch_size=32):
        super().__init__()
        self.train_seqs, self.test_seqs, self.val_seqs = train_seqs, test_seqs, val_seqs
        self.batch_size = batch_size
        self.max_len = max(len(X) for X, _ in train_seqs)
        self.train_dataset = PaddedSpeciesDataset(train_seqs, max_len=self.max_len)
        self.test_dataset = PaddedSpeciesDataset(test_seqs, max_len=self.max_len)
        self.val_dataset = PaddedSpeciesDataset(val_seqs, max_len=self.max_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)

