# Imports
# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset, DataLoader, random_split
import lmdb
import pickle
from torch_geometric.data import Batch
import torch
import time

# Program dataset

NUM_GRAPHS = 320

class OmnicsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset_path = "dataset"
        self.processed_path = "lmdb"
        
        datasetName = "data"

        self.lmdb_path = self.processed_path

        self._load_lmdb()


    def _load_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin() as txn:
            print("hello!")
            self.len =  pickle.loads(txn.get(str("len").encode()))
            self.input_dims = pickle.loads(txn.get(str("inputs").encode()))
            self.num_graphs = len(self.input_dims)
            
    def __getitem__(self, idx):
        with self.env.begin() as txn:
            item_bytes = txn.get(str(idx).encode())

        return pickle.loads(item_bytes)
    
    def __len__(self):
        return self.len
    
    def get_inputs_dims(self):
        return self.input_dims

    def get_num_graphs(self):
        return self.num_graphs


# Program Collate Function.

def collate(batch):
    # batch is: [([Data_0, Data_1, ..., Data_{k-1}], y, label), ...]
    graphs_list, ys, labels = zip(*batch)  # unzip once

    # Transpose: [N][K] -> [K][N], so we can batch position-wise
    # Each 'slot' is the i-th graph across the batch
    slots = zip(*graphs_list)

    batched_graphs = [Batch.from_data_list(slot) for slot in slots]

    y = torch.as_tensor(ys)  # avoids copy if already tensor-like

    return batched_graphs, y, labels


# main function with hyperparameters.


if  __name__ == "__main__":
    dataset = OmnicsDataset()
    batch_size = 32
    # try out the collate function with the batch thing.



    #Okay, so idea number one failed; the tallest element isn't always the adosrbate.

    # I can check to see if every element falls into one of these:

    # There is 1 hydrogen, 1 carbon, 1 oxygen.

    # what irriates me above all else is the idea that there could be one with multiple carbons, multiple oxygens, and 0 Hydrogens.

    # Okay, screw this. I can't handle this dudes. It's way to slow to process everything here



    split = random_split(dataset, [0.7, 0.2, 0.1])
    train_dataset = split[0]
    val_dataset = split[1]

    # alright, what do I want to do?
    # find the tallest one.
    # Check the identity.
    # If H, leave.
    # If O or C, then find the 2nd highest.
    # What's the easiest way? May be to duplicate. Have some sort of keyword.
    # No, I don't want to deal with that. 
    # Just process the whole dataset right here, before I toss it into the loader. Good.
    # Then toy with the collate function. 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, collate_fn=collate)

    print(time.time())
    for batch in train_loader:
        breakpoint()

    print(time.time())