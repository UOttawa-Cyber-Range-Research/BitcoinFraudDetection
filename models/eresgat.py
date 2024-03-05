import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional
import argparse

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
    if opt.model == 'eresgat':
        model = EResGAT(
            num_of_layers=5,
            num_heads_per_layer=[6, 6, 6, 6, 6],
            num_features_per_layer=[opt.input_dim, 8, 8, 8, 8, 8],
            num_identity_feats=8,
            device=opt.device,
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

class EResGAT(torch.nn.Module):
    def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, num_identity_feats,
                device, add_skip_connection=False, bias=True, residual=False, dropout=0.2):
        super().__init__()
        assert num_of_layers == len(num_heads_per_layer) == len(num_features_per_layer) - 1, f'Enter valid arch params.'

        # Define the instance variables
        self.device = device
        self.num_of_layers = num_of_layers
        num_heads_per_layer = [1] + num_heads_per_layer

        gat_layers = []
        for i in range(num_of_layers):
            layer = GATLayer(
                num_in_features=num_features_per_layer[i] * num_heads_per_layer[i] if i < 1 or residual == False else
                num_features_per_layer[i] * num_heads_per_layer[i] + num_features_per_layer[0],
                num_out_features=num_features_per_layer[i+1],
                num_of_heads=num_heads_per_layer[i+1],
                num_in_identity=num_features_per_layer[0],
                num_out_identity=num_identity_feats,
                concat=True if i < num_of_layers - 1 else False,
                activation=nn.ELU() if i < num_of_layers - 1 else None,
                dropout_prob=dropout,
                add_skip_connection=add_skip_connection,
                residual=residual,
                bias=bias
            )
            gat_layers.append(layer)

        self.gat_net = nn.Sequential(
            *gat_layers,
        )

    def forward(self, x, adj_t):
        # Define the adj list here
        self.adj_lists = defaultdict(set)
        
        # Define the edge index == total data points
        edge_idx = np.arange(adj_t.T.shape[0])
               
        # We have more modular data for graphs here
        self.adj = adj_t.T  # shape : (num_index, 2)
        
        # Write the code for generating the edge index for this set of graph
        for i, line in enumerate(self.adj):
            node1 = line[0].item()
            node2 = line[1].item()
            self.adj_lists[node1].add(i)
            self.adj_lists[node2].add(i)
            
        # Step 1: Sample 2-hop Neighborhood
        source_nodes_ids, target_nodes_ids = [], []
        seen_edges = set()
        edges = set(edge_idx)
        edges_neigh = set(edge_idx)

        for i in range(2):
            source_nodes_ids, target_nodes_ids, seen_edges, edges_neigh = self.build_edge_index(source_nodes_ids,
                                                                                                target_nodes_ids, seen_edges,
                                                                                                edges_neigh)
            edges = edges.union(edges_neigh)

        # Step 2: Construct Batch in_nodes_features
        in_nodes_features = x

        # Step 3: Construct new mapping; unique_map converts the edges to (0, len(edges))
        unique_map = {}
        for idx, edge in enumerate(edges):
            unique_map[edge] = idx
            
        source_nodes_ids = [unique_map[ids] for ids in source_nodes_ids]
        target_nodes_ids = [unique_map[ids] for ids in target_nodes_ids]

        # Step 4: Build edge_index; shape = (2, E), where E is the number of edges in the graph
        edge_index = np.row_stack((source_nodes_ids, target_nodes_ids))
        edge_index = torch.tensor(edge_index, dtype=torch.int64 ,device=self.device)

        # Step 5: Mapped edge_idx
        map_edge_idx = [unique_map[ids] for ids in edge_idx]
        map_edge_idx = torch.tensor(map_edge_idx, dtype=torch.int64, device=self.device)

        data = (in_nodes_features, in_nodes_features, edge_index, map_edge_idx)
        output, _ = self.gat_net(data)
        _, out, _, idx = output
        return out

    def build_edge_index(self, source_nodes_ids, target_nodes_ids, seen_edges, edges_neigh):
        new_neigh = set()
        for edge in edges_neigh:
            nodes = self.adj[edge]
            for node in nodes:
                neigh = self.adj_lists.get(node.item())
                new_neigh = new_neigh.union(neigh)
                for edge_neigh in neigh:
                    if (edge, edge_neigh) not in seen_edges and \
                            (edge_neigh, edge) not in seen_edges:
                        source_nodes_ids.append(edge)
                        target_nodes_ids.append(edge_neigh)
                        seen_edges.add((edge, edge_neigh))

        return source_nodes_ids, target_nodes_ids, seen_edges, new_neigh


class GATLayer(torch.nn.Module):
    src_nodes_dim = 0  # position of source nodes in edge index
    trg_nodes_dim = 1  # position of target nodes in edge index

    nodes_dim = 0  # node dimension (axis is maybe a more familiar term nodes_dim is the position of "N" in tensor)
    head_dim = 1  # attention head dim

    def __init__(self, num_in_features, num_out_features, num_of_heads, num_in_identity, num_out_identity,
                 concat=True, activation=nn.ELU(), dropout_prob=0.6, add_skip_connection=False, residual=True, bias=True):

        super().__init__()

        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = concat
        self.add_skip_connection = add_skip_connection
        self.residual = residual


        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        if not concat:
            if self.residual:
                self.linear_last = nn.Linear(num_of_heads * num_out_features + num_in_identity, 1, bias=True)
            else:
                self.linear_last = nn.Linear(num_of_heads * num_out_features, 1, bias=True)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, num_of_heads, num_out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(num_of_heads * num_out_features))
        else:
            self.register_parameter('bias', None)

        if add_skip_connection:
            self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)
        else:
            self.register_parameter('skip_proj', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_prob)
        self.init_params()

    def forward(self, data):
        identity_features, in_nodes_features, edge_index, map_edge_idx = data  # unpack data
        num_of_nodes = in_nodes_features.shape[self.nodes_dim]
        assert edge_index.shape[0] == 2, f'Expected edge index with shape=(2,E) got {edge_index.shape}'

        nodes_features_proj = self.linear_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)

        scores_source = (nodes_features_proj * self.scoring_fn_source).sum(dim=-1)
        scores_target = (nodes_features_proj * self.scoring_fn_target).sum(dim=-1)

        scores_source_lifted, scores_target_lifted, nodes_features_proj_lifted = self.lift(scores_source, scores_target,
                                                                                           nodes_features_proj,
                                                                                           edge_index)
        scores_per_edge = self.leakyReLU(scores_source_lifted + scores_target_lifted)

        attentions_per_edge = self.neighborhood_aware_softmax(scores_per_edge, edge_index[self.trg_nodes_dim],
                                                              num_of_nodes)
        attentions_per_edge = self.dropout(attentions_per_edge)

        nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge

        out_nodes_features = self.aggregate_neighbors(nodes_features_proj_lifted_weighted, edge_index,
                                                      in_nodes_features, num_of_nodes)

        out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features)
        if self.residual:
            out_nodes_features = torch.cat((out_nodes_features, identity_features), 1)

        if not self.concat: # final layer
            final_features = self.linear_last(out_nodes_features)
            return (identity_features, final_features, edge_index, map_edge_idx), out_nodes_features
        else:
            return identity_features, out_nodes_features, edge_index, map_edge_idx

    def neighborhood_aware_softmax(self, scores_per_edge, trg_index, num_of_nodes):
        """
        As the fn name suggest it does softmax over the neighborhoods. Example: say we have 5 nodes in a graph.
        Two of them 1, 2 are connected to node 3. If we want to calculate the representation for node 3 we should take
        into account feature vectors of 1, 2 and 3 itself. Since we have scores for edges 1-3, 2-3 and 3-3
        in scores_per_edge variable, this function will calculate attention scores like this: 1-3/(1-3+2-3+3-3)
        (where 1-3 is overloaded notation it represents the edge 1-3 and its (exp) score) and similarly for 2-3 and 3-3
         i.e. for this neighborhood we don't care about other edge scores that include nodes 4 and 5.

        Note:
        Subtracting the max value from logits doesn't change the end result but it improves the numerical stability
        and it's a fairly common "trick" used in pretty much every deep learning framework.
        Check out this link for more details:

        https://stats.stackexchange.com/questions/338285/how-does-the-subtraction-of-the-logit-maximum-improve-learning

        """
        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1 (this will improve the numerical stability)
        scores_per_edge = 3 * (scores_per_edge - scores_per_edge.max())/(scores_per_edge.max() - scores_per_edge.min())
        exp_scores_per_edge = scores_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_scores_neighborhood_aware(exp_scores_per_edge, trg_index,
                                                                                num_of_nodes)

        # 1e-16 is theoretically not needed but is only there for numerical stability (avoid div by 0) - due to the
        # possibility of the computer rounding a very small number all the way to 0.
        attentions_per_edge = exp_scores_per_edge / (neigborhood_aware_denominator + 1e-16)

        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise multiplication with projected node features
        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_scores_neighborhood_aware(self, exp_scores_per_edge, trg_index, num_of_nodes):
        # The shape must be the same as in exp_scores_per_edge (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_scores_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_scores_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)

        # position i will contain a sum of exp scores of all the nodes that point to the node i (as dictated by the
        # target index)
        neighborhood_sums.scatter_add_(self.nodes_dim, trg_index_broadcasted, exp_scores_per_edge)

        # Expand again so that we can use it as a softmax denominator. e.g. node i's sum will be copied to
        # all the locations where the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighborhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbors(self, nodes_features_proj_lifted_weighted, edge_index, in_nodes_features, num_of_nodes):
        size = list(nodes_features_proj_lifted_weighted.shape)  # convert to list otherwise assignment is not possible
        size[self.nodes_dim] = num_of_nodes  # shape = (N, NH, FOUT)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)

        # shape = (E) -> (E, NH, FOUT)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[self.trg_nodes_dim],
                                                        nodes_features_proj_lifted_weighted)
        # aggregation step - we accumulate projected, weighted node features for all the attention heads
        # shape = (E, NH, FOUT) -> (N, NH, FOUT)
        out_nodes_features.scatter_add_(self.nodes_dim, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

        return out_nodes_features

    def lift(self, scores_source, scores_target, nodes_features_matrix_proj, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E (that's where the "lift" comes from).
        """
        src_nodes_index = edge_index[self.src_nodes_dim]
        trg_nodes_index = edge_index[self.trg_nodes_dim]

        # Using index_select is faster than "normal" indexing (scores_source[src_nodes_index]) in PyTorch!
        scores_source = scores_source.index_select(self.nodes_dim, src_nodes_index)
        scores_target = scores_target.index_select(self.nodes_dim, trg_nodes_index)
        nodes_features_matrix_proj_lifted = nodes_features_matrix_proj.index_select(self.nodes_dim, src_nodes_index)

        return scores_source, scores_target, nodes_features_matrix_proj_lifted

    def explicit_broadcast(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    def init_params(self):
        """
        The reason we're using Glorot (aka Xavier uniform) initialization is because it's a default TF initialization:
            https://stackoverflow.com/questions/37350131/what-is-the-default-variable-initializer-in-tensorflow

        The original repo was developed in TensorFlow (TF) and they used the default initialization.
        Feel free to experiment - there may be better initializations depending on your problem.

        """
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if self.add_skip_connection:  # add skip or residual connection
            if out_nodes_features.shape[-1] == in_nodes_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_nodes_features += in_nodes_features.unsqueeze(1)
            else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads,
                                                                             self.num_out_features)

        out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)

        if self.bias is not None:
            out_nodes_features += self.bias

        return out_nodes_features if self.activation is None else self.activation(out_nodes_features)