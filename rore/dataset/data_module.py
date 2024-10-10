import pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
logger = logging.getLogger(__name__)

class DocumentDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset,
                 test_dataset,
                 batch_size,
                 val_test_batch_size=None,
                 pin_memory=False,
                 persistent_workers=True,
                 preprocess_workers=8,
                 shuffle=True,
                 **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        if self.val_test_batch_size is None: self.val_test_batch_size = self.batch_size

    def train_dataloader(self):
        if self.train_dataset is not None:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=self.shuffle,
                num_workers=self.preprocess_workers,
            )
        else: return None

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.val_test_batch_size,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=False,
                num_workers=self.preprocess_workers,
            )
        else: return None

    def test_dataloader(self):
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_test_batch_size,
                pin_memory=self.pin_memory,
                persistent_workers=self.persistent_workers,
                shuffle=False,
                num_workers=self.preprocess_workers,
            )
        else: return None

    def predict_dataloader(self):
        return self.test_dataloader()