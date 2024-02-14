import torch
import argparse
import numpy as np
from typing import Any, Dict, Optional
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GPSConv, BatchNorm
from torch_geometric.nn.norm import GraphNorm
from torch.nn import (
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
                
class GPS(torch.nn.Module):
    def __init__(self, channels: int, num_layers: int,
                attn_type: str, attn_kwargs: Dict[str, Any]):
        super().__init__()
        
        # Define the init transform
        self.init_transform = torch.nn.Linear(channels, 64)
        channels = 64
        
        # Define the gps layers
        self.convs = ModuleList()
        self.dropout = attn_kwargs["dropout"]
        for _ in range(num_layers):
            # Define the sequential model for the GPS layer
            nn = Sequential(
                Linear(channels, channels),
                ReLU(),
                Linear(channels, channels),
            )
            
            # Define the conv layer
            conv = GPSConv(channels, GINConv(nn), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            
            # Append the conv layer to the list
            self.convs.append(conv)

        self.mlp = Sequential(
            Linear(channels, channels // 2),
            ReLU(),
            Linear(channels // 2, channels // 4),
            ReLU(),
            Linear(channels // 4, 1),
        )
        
        self.mpl_x = Linear(channels, channels)
        self.bns = torch.nn.ModuleList(GraphNorm(channels) for _ in range(num_layers))

    def forward(self, x, edge_index):
        # Convert the initial transform
        x = self.init_transform(x)
        
        # Run the conv and get the output
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        
        # Return the model
        return self.mlp(x)
    
    
def build_model(
    opt, 
    model_only: bool = False,
    class_weights: Optional[np.ndarray] = None,
):
    attn_kwargs = {'dropout': 0.2}
    if opt.model == 'gps':
        model = GPS(
            channels=opt.input_dim,
            num_layers=opt.num_layers,
            attn_type="multihead",
            attn_kwargs=attn_kwargs,
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


def args(parser: argparse.ArgumentParser = argparse.ArgumentParser()):
    try:
        # add stuff to parser
        parser.add_argument('-num_layers', type=int, default=4)
        parser.add_argument('-pe_dim', type=int, default=8)
        parser.add_argument('-class_weight', type=int, default=10)
        parser.add_argument('-lr', type=float, default=1e-3)
    except argparse.ArgumentError:
        # jupyter sometimes remembers the parse, in this case just return
        pass

    return parser