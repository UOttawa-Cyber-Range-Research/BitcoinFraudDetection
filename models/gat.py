
# GAT setup taken from torch_geometric demo
# https://medium.com/stanford-cs224w/applying-graph-ml-to-classify-il-licit-bitcoin-transactions-fd32a1ff5dab

import torch.nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import numpy as np
import argparse

def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=3)
        parser.add_argument('-hidden_dim', type=int, default=128)
        parser.add_argument('-heads', type=int, default=2)
        parser.add_argument('-dropout', type=float, default=0.5)
        parser.add_argument('-lr', type=float, default=0.01)
        parser.add_argument('-class_weight', type=int, default=10)
    except argparse.ArgumentError:
        # jupyter sometimes remembers the parse, in this case just return
        pass

    return parser

class GAT(torch.nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        output_dim, 
        heads, num_layers,
        dropout, 
        emb=False
    ):
        super().__init__()
        conv_model = GATConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            conv_model(
                input_dim, hidden_dim, 
                heads=heads
            ))
        assert (
            num_layers >= 1
        ), 'Number of layers is not >=1'
        for _ in range(num_layers-1):
            self.convs.append(
                conv_model(
                    heads * hidden_dim, 
                    hidden_dim, 
                    heads=heads
                ))

        # post-message-passing
        self.post_mp = torch.nn.Sequential(
            torch.nn.Linear(
                heads * hidden_dim, hidden_dim
            ), 
            torch.nn.Dropout(dropout), 
            torch.nn.Linear(
                hidden_dim, output_dim
            ))

        self.dropout = dropout
        self.num_layers = num_layers

        self.emb = emb

    def forward(self, x, adj_t):          
        for i in range(self.num_layers):
            x = self.convs[i](x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout,training=self.training)

        x = self.post_mp(x)

        if self.emb == True:
            return x

        return torch.sigmoid(x)

from typing import Optional

def build_model(
    opt, 
    model_only: bool = False,
    class_weights: Optional[np.ndarray] = None,
):

    if opt.model == 'gat':
        model = GAT(
            input_dim=opt.input_dim,
            hidden_dim=opt.hidden_dim,
            output_dim=1,
            heads=opt.heads,
            num_layers=opt.num_layers,
            dropout=opt.dropout,
            emb=True,
        ).to(opt.device)
    else:
        raise NotImplementedError

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
