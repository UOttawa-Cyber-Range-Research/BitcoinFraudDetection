import torch.nn
from torch_geometric.nn import ResGatedGraphConv, BatchNorm
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
import numpy as np
import argparse

def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=3)
        parser.add_argument('-hidden_dim', type=int, default=128)
        parser.add_argument('-dropout', type=float, default=0.2)
        parser.add_argument('-lr', type=float, default=1e-3)
        parser.add_argument('-class_weight', type=int, default=10)
        parser.add_argument('-norm')
    except argparse.ArgumentError:
        pass

    return parser

class ResGatedGCN(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers, 
            dropout,
            norm,
        ):

        super().__init__()
        self.convs = torch.nn.ModuleList(
            [ResGatedGraphConv(input_dim, hidden_dim)]
            + [ResGatedGraphConv(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
            + [ResGatedGraphConv(hidden_dim, output_dim)]
        )
        
        if norm == "GN":
            self.bns = torch.nn.ModuleList(GraphNorm(hidden_dim) for _ in range(num_layers - 1))
        else:
            self.bns = torch.nn.ModuleList(BatchNorm(hidden_dim) for _ in range(num_layers - 1))
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

    if opt.model == 'rggcn':
        model = ResGatedGCN(
            input_dim=opt.input_dim,
            hidden_dim=opt.hidden_dim,
            output_dim=7,
            num_layers=opt.num_layers,
            dropout=opt.dropout,
            norm=opt.norm,
        ).to(opt.device)
    else:
        raise NotImplementedError

    model.reset_parameters()

    if model_only:
        return model

    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=opt.lr,
        weight_decay=5e-3,
    )
    loss_fn = torch.nn.CrossEntropyLoss(
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
