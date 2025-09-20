import muon as mu
import os
import lmdb
import pandas as pd
import numpy as np
import pickle
from torch_geometric.data import Data
import torch

# Read the entire H5MU file into a MuData object
mdata = mu.read("gse129705.h5mu")
adata = mdata.mod["mrna"]


# get resposnes
dataframe = adata.obs


inputs = adata.X

class_dict = {'MONTH 3': 0, 'BASELINE':1}


gene_sym_to_num = dict()
all_genes = adata.var['genesym'].tolist()
all_gene_set = set(all_genes)
for i, gene in enumerate(adata.var['genesym']):
    gene_sym_to_num[gene] = i



# process graphs

graphs_list = os.listdir("./processed")
graphs_list.sort()

gene_matrix = []

all_edges = []
all_masks = []

for graph in graphs_list:
    folder = "./processed/{}".format(graph)
    edge_path = "{}/edges.tsv".format(folder)
    node_path = "{}/nodes.tsv".format(folder)



    df = pd.read_csv(edge_path, sep='\t')

    node_df = pd.read_csv(node_path, sep='\t')

    # what do I want? A list of genes, a matrix of gene data.

    edges = [[],[]]
    nodes = set()
    for i in range(df.shape[0]):

        source = df.iloc[i]['source'].item()
        dest = df.iloc[i]['target'].item()
        edges[0].append(source)
        edges[1].append(dest)
        nodes.add(source)
        nodes.add(dest)
    
    nodes = [n for n in nodes]
    nodes.sort()

    node_index_dict = dict()

    # make the edges the smallest number possible
    for i, node in enumerate(nodes):
        node_index_dict[node] = i
    edges[0] = [node_index_dict[n] for n in edges[0]]
    edges[1] = [node_index_dict[n] for n in edges[1]]

    node_gene_map = dict()

    genes = set()

    for i in range(node_df.shape[0]):
        node = node_df.iloc[i]

        id = node['id'].item()

        if id not in nodes:
            continue

        node_genes = node['name'].replace("...","").split(", ")
        node_genes = [n for n in node_genes if n in all_gene_set]
        node_genes.sort()

        node_gene_map[id] = node_genes
        genes.update(node_genes)

    if(len(genes) == 0):
        print("Bad: {}".format(graph))
        continue

    genes = [g for g in genes]
    genes.sort()


    # make the masks
    masks = [[False] * len(genes) for i in range(len(nodes))]


    for i, n in enumerate(nodes):
        node_genes = node_gene_map[n]
        indicies = [genes.index(g) for g in node_genes]

        for index in indicies:
            masks[i][index] = True
    
    all_edges.append(torch.tensor(edges))
    gene_matrix.append(genes)
    all_masks.append(masks)


# I want to end with a list, and a mask for each node.
# nodes and genes should be sorted alphabetically.
# screw memory for now.
lmdb_path = "./lmdb"
os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)
env = lmdb.open(lmdb_path, map_size=2 * 10**9)  # Adjust map_size as needed
with env.begin(write=True) as txn:

    for i in range(len(adata)):
        input = inputs[i].tolist()
        response = class_dict[dataframe.iloc[i]['response']]
        id = dataframe.iloc[i]['unique_id']
        X = []

        for g in range(len(all_masks)):
            gene_indicies = [all_genes.index(gene) for gene in gene_matrix[g]]
            gene_values = [input[index] for index in gene_indicies]
            cur_gene_matrix = []

            masks = all_masks[g]

            for mask in masks:
                cur_genes = np.array([_ for _ in gene_values])
                mask = np.array(mask)
                cur_genes[~mask] = 0
                cur_gene_matrix.append(cur_genes.tolist())

            X.append(torch.tensor(cur_gene_matrix))
        
        input_graphs = []
        for graph_num in range(len(X)):
            graph = Data(x=X[graph_num], edge_index = all_edges[graph_num])
            input_graphs.append(graph)
        
        print(i)
        txn.put(str(i).encode(), pickle.dumps((input_graphs, response, id)))
    
    txn.put(str("len").encode(), pickle.dumps(len(adata)))
    txn.put(str("num_graphs").encode(), pickle.dumps(len(all_edges)))

# okay, I need to make the graphs next:

# some of them have no edges.

# go through each. 

# for each grpah:

"""
Check if has edges. If not, discard.

If so, store the edge adjacency.

Then, get the list of graphs in each.

Make a list of the genes etc. that makes up the vector. 

Make a mask for each.

Store each vector for each node, with masks applied.

Then, finally put in the giant list of matricies, the output, and the label.
"""