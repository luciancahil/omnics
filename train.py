# Imports
# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset, DataLoader, random_split
import lmdb
import pickle
from torch_geometric.data import Batch
import torch
import time
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
from torch.nn import Linear, Dropout, Softmax
# Program dataset


class GraphNet(nn.Module):
    def __init__(self, input_dims, num_classes=2, hidden_dim=16, hidden_layers=8, hidden_dropout=0.5):
        super(GraphNet, self).__init__()
        self.graph_layer = nn.ModuleList([GATConv(input_dim, hidden_dim) for input_dim in input_dims])
        self.num_graphs = len(self.graph_layer) 
        self.post_graph_conv = nn.ModuleList([Linear(hidden_dim, 1) for _ in self.graph_layer])


        self.pooled_convs = [nn.Sequential(Linear(self.num_graphs, hidden_dim), Dropout(hidden_dropout))]

        self.pooled_convs.extend([nn.Sequential(Linear(hidden_dim, hidden_dim), Dropout(hidden_dropout))
                                  for _ in range(hidden_layers)])
        
        self.pooled_convs.append(Linear(hidden_dim, num_classes))

        self.pooled_convs = nn.ModuleList(self.pooled_convs)

        self.softMax = Softmax()


    def forward(self, input_graphs):
        mlp_input = []
        # once said and done, this should be batch_size * num_graphs

        # starts off this way, because we want to transpose
        pooled = torch.zeros((0, input_graphs[0].ptr.shape[0]-1))

        for i, graph in enumerate(input_graphs):
            # run through 
            graph_output = (self.graph_layer[i](graph.x, graph.edge_index))
            graph_output = self.post_graph_conv[i](graph_output)
            graph_output = graph_output.squeeze()
            pointers = graph.ptr
            cur_pooled = []


            for p in range(len(pointers[:-1])):
                start = pointers[p]
                end = pointers[p + 1]

                cur_pooled.append(torch.mean(graph_output[start:end]))

            cur_pooled = torch.tensor(cur_pooled).unsqueeze(0)
            pooled = torch.cat((pooled, cur_pooled), dim=0)
        

        pooled = pooled.T

        for layer in self.pooled_convs:
            pooled = layer(pooled)
        
        return pooled

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
    
    def get_input_dims(self):
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


def main(batch_size = 32, split_proportions=[0.7, 0.2, 0.1]):
    dataset = OmnicsDataset()

    # try out the collate function with the batch thing.



    #Okay, so idea number one failed; the tallest element isn't always the adosrbate.

    # I can check to see if every element falls into one of these:

    # There is 1 hydrogen, 1 carbon, 1 oxygen.

    # what irriates me above all else is the idea that there could be one with multiple carbons, multiple oxygens, and 0 Hydrogens.

    # Okay, screw this. I can't handle this dudes. It's way to slow to process everything here



    split = random_split(dataset, split_proportions)
    train_dataset = split[0]
    val_dataset = split[1]
    test_dataset = split[2]

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
    test_loader = DataLoader(test_dataset, batch_size = batch_size, collate_fn=collate)

    model = GraphNet(dataset.get_input_dims())

    for batch in train_loader:
        inputs, targets, labels = batch
        outputs = model(inputs)
        loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(outputs, targets)

        breakpoint()


if  __name__ == "__main__":
    main()
