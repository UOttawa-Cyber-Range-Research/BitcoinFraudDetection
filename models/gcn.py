
# GCN setup taken from torch_geometric demo
# https://medium.com/stanford-cs224w/applying-graph-ml-to-classify-il-licit-bitcoin-transactions-fd32a1ff5dab

import torch.nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import numpy as np
import argparse

def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=3)
        parser.add_argument('-hidden_dim', type=int, default=128)
        parser.add_argument('-dropout', type=float, default=0.1)
        parser.add_argument('-lr', type=float, default=0.01)
        parser.add_argument('-class_weight', type=int, default=10)
    except argparse.ArgumentError:
        # jupyter sometimes remembers the parse, in this case just return
        pass

    return parser

class GCN(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers, 
            dropout,
        ):

        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim, hidden_dim)]
            + [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
            + [GCNConv(hidden_dim, output_dim)]
        )
        self.bns = torch.nn.ModuleList(torch.nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i in range(len(self.convs)):
            x = self.convs[i](x, adj_t)
            if i < len(self.convs) - 1:
                x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                return x

from typing import Optional

def build_model(
    opt, 
    model_only: bool = False,
    class_weights: Optional[np.ndarray] = None,
):

    if opt.model == 'gcn':
        model = GCN(
            input_dim=opt.input_dim,
            hidden_dim=opt.hidden_dim,
            output_dim=1,
            num_layers=opt.num_layers,
            dropout=opt.dropout,
        ).to(opt.device)
    else:
        raise NotImplementedError

    model.reset_parameters()

    if model_only:
        return model

    optimizer = torch.optim.Adam(
        model.parameters(), 
        opt.lr,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(class_weights),
    ).to(opt.device)

    return model, optimizer, loss_fn

from utils import restore_args
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
