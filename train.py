# Imports
# getitem must return: None, input_graph, target_graph, null, null.
import os
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import lmdb
import pickle
from torch_geometric.data import Batch
import torch
import time
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, TAGConv, knn
from torch.nn import Linear, Dropout, Softmax, LeakyReLU
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool

# Program dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOSS_FN = nn.CrossEntropyLoss()

class GraphNet(nn.Module):
    def __init__(self, input_dims, num_classes=2, hidden_dim=16, hidden_layers=8, hidden_dropout=0.0):
        super(GraphNet, self).__init__()
        self.graph_layer = nn.ModuleList([GATConv(input_dim, hidden_dim) for input_dim in input_dims])
        self.num_graphs = len(self.graph_layer) 
        self.post_graph_conv = nn.ModuleList([Linear(hidden_dim, 1) for _ in self.graph_layer])


        self.pooled_convs = [nn.Sequential(Linear(self.num_graphs, hidden_dim), Dropout(hidden_dropout), LeakyReLU())]

        self.pooled_convs.extend([nn.Sequential(Linear(hidden_dim, hidden_dim), Dropout(hidden_dropout), LeakyReLU())
                                  for _ in range(hidden_layers)])
        
        self.pooled_convs.append(Linear(hidden_dim, num_classes))

        self.pooled_convs = nn.ModuleList(self.pooled_convs)


    def forward(self, input_graphs):
        device = input_graphs[0].x.device
        num_graphs_in_batch = input_graphs[0].ptr.numel() - 1
        pooled = torch.empty((0, num_graphs_in_batch), device=device)

        for i, graph in enumerate(input_graphs):
            out = self.graph_layer[i](graph.x, graph.edge_index)        # [num_nodes, hidden]
            out = self.post_graph_conv[i](out).squeeze(-1)              # [num_nodes]
            cur_pooled = global_mean_pool(out, graph.batch)             # [B]
            cur_pooled = cur_pooled.unsqueeze(0)
            pooled = torch.cat((pooled, cur_pooled), dim=0)

        pooled = pooled.T  # [B, num_input_graphs]
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
            item_bytes = txn.get(str(int(idx)).encode())

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


def train_epoch(epoch_type, epoch, model, loader, regularization_lambda, optimizer=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    num_samples = 0

    for batch in loader:
        if is_train:
            optimizer.zero_grad()
            loss, acc, bs, correct = get_loss_and_acc(model, batch, regularization_lambda)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                loss, acc, bs, correct = get_loss_and_acc(model, batch)

        num_samples += bs
        total_loss += float(loss.item()) * bs
        total_correct += int(correct)

    avg_loss = total_loss / max(1, num_samples)
    avg_acc  = total_correct / max(1, num_samples)
    print(f"{epoch_type} Epoch {epoch}: Loss = {avg_loss:.4f} and Accuracy = {avg_acc:.4f}")
    return avg_loss, avg_acc

def get_reg_loss(model):
    ann_first_layer = model.pooled_convs[0][0]
    parameters = [p for p in ann_first_layer.parameters()]
    weights = parameters[0]

    return torch.norm(weights, p=2, dim=0).sum()

def get_loss_and_acc(model, batch, lam):
    inputs, targets, labels = batch

    # Move every Batch to DEVICE
    inputs = [g.to(DEVICE, non_blocking=True) for g in inputs]
    # Move targets
    targets = targets.to(DEVICE, non_blocking=True).long()

    logits = model(inputs)
    targets = targets.to(logits.device).long()
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    acc_loss = LOSS_FN(logits, targets)
    acc = correct / logits.size(0)

    reg_loss = get_reg_loss(model)

    loss = acc_loss + lam * reg_loss


    return loss, acc, logits.size(0), correct


def kfold_split(dataset, k=5, seed=42):
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold_size = n // k
    folds = []
    for i in range(k):
        val_idx = indices[i*fold_size : (i+1)*fold_size]
        train_idx = torch.cat([indices[:i*fold_size], indices[(i+1)*fold_size:]])
        folds.append((train_idx, val_idx))
    return folds

def main( lr=0.001, hidden_dim=16, hidden_layers=8, hidden_dropout=0.0, weight_decay=1e-6, regularization_lambda=1e-6):
    dataset = OmnicsDataset()
    max_epochs = 100
    batch_size = 32
    # try out the collate function with the batch thing.



    #Okay, so idea number one failed; the tallest element isn't always the adosrbate.

    # I can check to see if every element falls into one of these:

    # There is 1 hydrogen, 1 carbon, 1 oxygen.

    # what irriates me above all else is the idea that there could be one with multiple carbons, multiple oxygens, and 0 Hydrogens.

    # Okay, screw this. I can't handle this dudes. It's way to slow to process everything here


    split_proportions=[0.9, 0.1]
    split = random_split(dataset, split_proportions,  generator=torch.Generator().manual_seed(42))
    train_val_dataset = split[0]
    test_dataset = split[1]
    cross_validation_splits = 5

    folds = kfold_split(train_val_dataset, cross_validation_splits)
    
    best_val_list = []

    # alright, what do I want to do?
    # find the tallest one.
    # Check the identity.
    # If H, leave.
    # If O or C, then find the 2nd highest.
    # What's the easiest way? May be to duplicate. Have some sort of keyword.
    # No, I don't want to deal with that. 
    # Just process the whole dataset right here, before I toss it into the loader. Good.
    # Then toy with the collate function. 

    for (fold_num, (train_idx, val_idx)) in enumerate(folds):
        print("Fold: {}".format(fold_num))
        best_acc = 0
        train_dataset = Subset(dataset, train_idx)
        val_dataset   = Subset(dataset, val_idx)        

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=collate, pin_memory=torch.cuda.is_available())
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                                collate_fn=collate, pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                                collate_fn=collate, pin_memory=torch.cuda.is_available())

        model = GraphNet(dataset.get_input_dims(), num_classes=2, hidden_dim=hidden_dim, hidden_layers=hidden_layers, hidden_dropout=hidden_dropout)
        model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer)
        model.train()

        for i in range(max_epochs):
            _,_ = train_epoch("Train", i, model, train_loader, regularization_lambda * float(i) / max_epochs, optimizer)
            if i % 5 == 0:
                val_loss, val_acc = train_epoch("Validation", i, model, val_loader, regularization_lambda * i / max_epochs, optimizer=None)
                scheduler.step(val_loss)
                if(val_acc > best_acc):
                    best_acc = val_acc
        
        best_val_list.append(best_acc)

    # return 1 - loss because BO finds the minimum
    return 1 - sum(best_val_list) / len(best_val_list)
        

if  __name__ == "__main__":
    main()
