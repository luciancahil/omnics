# Imports
# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset
import lmdb
import pickle

# Program dataset


class OmnicsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = "dataset"
        self.processed_path = "lmdb"
        
        datasetName = "data"
        self.len = 128

        self.lmdb_path = self.processed_path

        self._load_lmdb()


    def _load_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            print("hello!")

    def __getitem__(self, idx):
        with self.env.begin() as txn:
            item_bytes = txn.get(str(idx).encode())
            if item_bytes is None:
                return None, None, None, None, None
        return pickle.loads(item_bytes)

# Program Collate Function.

# 1 function for both train and eval.

# main function with hyperparameters.

dataset = OmnicsDataset()

print(dataset[0])