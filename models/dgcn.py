
import torch
import torch.nn as nn
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.nn.norm import GraphNorm
import torch.nn.functional as F
import numpy as np
import argparse

def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=4)
        parser.add_argument('-hidden_dim', type=int, default=128)
        parser.add_argument('-dropout', type=float, default=0.2)
        parser.add_argument('-lr', type=float, default=1e-3)
        parser.add_argument('-class_weight', type=int, default=10)
    except argparse.ArgumentError:
        pass

    return parser

class DeeperGCN(torch.nn.Module):
    def __init__(
            self, 
            input_dim, 
            hidden_dim, 
            output_dim, 
            num_layers, 
            dropout,
        ):

        super().__init__()
        
        # Not used #################
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Make the module list
        self.layer_list = nn.ModuleList()
        for _ in range(num_layers):
            # Define the conv layers
            conv = GENConv(in_channels=input_dim, out_channels=input_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            
            # Normalization and activation
            norm = GraphNorm(input_dim)
            act = nn.ReLU(inplace=True)
            
            # Define the deeper layers
            lyr = DeepGCNLayer(conv=conv, norm=norm, act=act, dropout=dropout)
            
            # Add to module list
            self.layer_list.append(lyr)
            
        # Define a final transformation layer
        self.linear_lyr = nn.Linear(input_dim, 1)

    def forward(self, x, adj_t):
        # Loop over the layers
        for lyr in self.layer_list:
            # Pass and fetch output
            x = lyr(x, adj_t)
            
        # Pass through transformation layer
        return self.linear_lyr(x)
            

from typing import Optional

def build_model(
    opt, 
    model_only: bool = False,
    class_weights: Optional[np.ndarray] = None,
):

    if opt.model == 'dgcn':
        model = DeeperGCN(
            input_dim=opt.input_dim,
            hidden_dim=opt.hidden_dim,
            output_dim=1,
            num_layers=opt.num_layers,
            dropout=opt.dropout,
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
