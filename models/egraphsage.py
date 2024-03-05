import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import argparse
from typing import Optional
from utils import restore_args
from collections import defaultdict

def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=3)
        parser.add_argument('-hidden_dim', type=int, default=128)
        parser.add_argument('-heads', type=int, default=4)
        parser.add_argument('-dropout', type=float, default=0.2)
        parser.add_argument('-lr', type=float, default=1e-3)
        parser.add_argument('-class_weight', type=int, default=10)
    except argparse.ArgumentError:
        # jupyter sometimes remembers the parse, in this case just return
        pass

    return parser

def build_model(
    opt, 
    model_only: bool = False,
    class_weights: Optional[np.ndarray] = None,
):
    if opt.model == 'egraphsage':
        model = EGraphSage(
            num_classes=1,
            residual=True,
            feature_shape=opt.input_dim,
        ).to(opt.device)
    else:
        raise NotImplementedError

    if model_only:
        return model

    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=opt.lr,
        weight_decay=5e-3,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights),
    ).to(opt.device)

    return model, optimizer, loss_fn

def load_model(
    opt, model_path: str, 
    item='state_dict',
    others=['scaler', 'edge_norm'],
):

    d = torch.load(model_path, map_location=opt.device)
    opt = restore_args(
        opt,
        d['opt'], 
    )
    checkpoint = d[item]

    model = build_model(opt, model_only=True)
    model.load_state_dict(checkpoint)

    return model, {
        k:d[k] for k in others
    }


class EGraphSage(nn.Module):
    def __init__(self, num_classes, residual, feature_shape):
        super(EGraphSage, self).__init__()
        # Define the class variables
        self.residual = residual

        if self.residual:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, 2 * 64 + feature_shape))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(num_classes, 2 * 64))
        init.xavier_uniform_(self.weight)

    def forward(self, x, adj_t):
        # Fill the global variable
        self.adj = adj_t.T.cpu().numpy()
        
        # Define the node mapping
        self.node_map = {}
        for i, node in enumerate(torch.unique(adj_t)):
            self.node_map[node.item()] = i
            
        # Define the adj list
        adj_lists = defaultdict(set)
        for i, line in enumerate(self.adj):
            node1 = self.node_map[line[0].item()]
            node2 = self.node_map[line[1].item()]
            adj_lists[node1].add(i)
            adj_lists[node2].add(i)
            
        # Define some other stuff
        num_nodes = len(torch.unique(adj_t))
        node_feat = np.ones((num_nodes, 64))
        node_features = nn.Embedding(node_feat.shape[0], node_feat.shape[1])
        node_features.weight = nn.Parameter(torch.from_numpy(node_feat).float(), requires_grad=False)
        self.edge_features = x.cpu()
        self.edge_features_ = nn.Embedding(x.shape[0], x.shape[1])
        self.edge_features_.weight = nn.Parameter(x.cpu().float(), requires_grad=False)
            
        # Define the encoders and aggregators
        agg1 = MeanAggregator(self.edge_features_, gcn=False, cuda=False)
        enc1 = Encoder(node_features, x.shape[1], 64, adj_lists,
                        agg1, num_sample=8, gcn=True, cuda=False)
        agg2 = MeanAggregator(self.edge_features_, gcn=False, cuda=False)
        self.enc = Encoder(lambda nodes: enc1(nodes).t(), x.shape[1], 64,
                   adj_lists, agg2, num_sample=8, base_model=enc1, gcn=True, cuda=False)
        
        # E-GraphSAGE
        edges = np.arange(adj_t.shape[-1])
        nodes = self.adj[edges]
        nodes = [set(z) for z in nodes]
        nodes = [i.item() for i in list(set.union(*nodes))]
        nodes_id = [self.node_map[node] for node in nodes]

        # construct mapping from node_id to index
        unique_map = {}
        for idx, id in enumerate(nodes_id):
            unique_map[id] = idx
        
        # Get the node embeds
        node_embeds = self.enc(nodes_id).t().cpu().detach().numpy() # (N,e)  e: embed_dim
        
        # Collect the embeds
        if self.residual:
            edge_embeds = np.array([np.concatenate(
                (node_embeds[unique_map[self.node_map[self.adj[edge][0]]]],
                 node_embeds[unique_map[self.node_map[self.adj[edge][1]]]],
                 self.edge_features[edge])) for edge in edges])
        else:
            edge_embeds = np.array([np.concatenate(
                (node_embeds[unique_map[self.node_map[self.adj[edge][0]]]],
                 node_embeds[unique_map[self.node_map[self.adj[edge][1]]]])) for edge in edges])
            
        edge_embeds = torch.FloatTensor(edge_embeds)
        scores = self.weight.mm(edge_embeds.t().cuda()) # W * embed: (c,2e) * (2e,E)  --> (c,E)  c : num of classes
        return scores.t()


class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighboring edges' embeddings
    """

    def __init__(self, edge_features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        edge_features -- function mapping LongTensor of edge ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.edge_features = edge_features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=None):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbor edges for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """

        _set = set  # Local pointers to functions (speed hack)
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [
                _set(_sample(
                    to_neigh,
                    num_sample,
                )) if len(to_neigh) >= num_sample else to_neigh
                for to_neigh in to_neighs
            ]  # sample = neighbor, if neighboorhood size is less than num_sample
        else:
            samp_neighs = to_neighs

        if self.gcn:  # gcn=True, add self node into neighbor
            samp_neighs = [
                samp_neigh.union(set([nodes[i]]))
                for i, samp_neigh in enumerate(samp_neighs)
            ]

        unique_edges_list = list(set.union(*samp_neighs))
        unique_edges = {n: i for i, n in enumerate(unique_edges_list)}
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_edges)))
        column_indices = [
            unique_edges[n] for samp_neigh in samp_neighs for n in samp_neigh
        ]

        row_indices = [
            i for i in range(len(samp_neighs))
            for j in range(len(samp_neighs[i]))
        ]

        mask[row_indices, column_indices] = 1

        if self.cuda:
            mask = mask.cuda()

        num_neigh = mask.sum(1, keepdim=True)  # torch.sum()  (n,m) --> (n, 1)
        mask = mask.div(num_neigh)  # normalization

        if self.cuda:
            embed_matrix = self.edge_features(
                torch.LongTensor(unique_edges_list).cuda())
        else:
            embed_matrix = self.edge_features(torch.LongTensor(unique_edges_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats


class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self,
                 node_features,
                 feature_dim,
                 embed_dim,
                 adj_lists,
                 aggregator,
                 num_sample=None,
                 base_model=None,
                 gcn=False,
                 cuda=False):
        super(Encoder, self).__init__()

        self.node_features = node_features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn  # True: Mean-agg  False: GCN-agg
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
            torch.FloatTensor(
                embed_dim, self.feat_dim + self.embed_dim if self.gcn else self.feat_dim
            )
        )
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        if type(nodes) != list:
            nodes = nodes.tolist()
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                                              self.num_sample)  # (nodes，neighbor edges，num samples)
        if self.gcn:
            if self.cuda:
                self_feats = self.node_features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.node_features(torch.LongTensor(nodes)) # (n , f) n : num of nodes
            neigh_feats[torch.isnan(neigh_feats)] = 1e-2
            combined = torch.cat([self_feats, neigh_feats], dim=1)  # (n , 2f)
        else:
            combined = neigh_feats
        combined = self.weight.matmul(combined.t())
        combined = F.relu(combined) # relu W* F^T : (e, 2f) * (2f, n)  --> (e,n)  e: embed_dim
        return combined