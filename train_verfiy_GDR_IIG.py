import os
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.simplefilter("ignore")

import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Optional
from metrics.IIG import *
from metrics.GDR import *
from utils import *
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor
from torch_geometric.utils import remove_self_loops, add_self_loops

# ----------------------------------------------------------
# ARGS
# ----------------------------------------------------------
def common_args():
    import argparse
    parser = argparse.ArgumentParser("Train graphical models for bitcoin data")

    parser.add_argument('model') # first positional argument
    parser.add_argument('-model_path', default='data/models')
    parser.add_argument('-rwpe', default="false")
    parser.add_argument('-walk_length', default=21)
    parser.add_argument('-model_accum_grads', type=int, default=1)
    parser.add_argument('-data_path', default='data/dataset')
    parser.add_argument(
        '-features_dir', 
        default="data/tables",
        help="""dir where features are from"""
    )
    parser.add_argument('-data_disable_semi_sup', action='store_true', default=False)
    parser.add_argument('-resample_semi_sup', default=None)
    parser.add_argument('-resample_factor_semi_sup', type=int, default=None)
    parser.add_argument('-additional_features', nargs='*')
    parser.add_argument('-train', action='store_false', default=True)
    parser.add_argument('-train_best_metric', default='bacc')
    parser.add_argument('-epochs', type=int, default=12)
    parser.add_argument('-tensorboard', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-refresh_cache', action='store_true', default=False)
    parser.add_argument('-norm', default="GN")
    return parser

def parse_args(
    parser: argparse.ArgumentParser,
    args: Optional[List[str]] = None,
):

    opt = parser.parse_args(args)

    if torch.cuda.is_available():
        opt.device = torch.device("cuda")
    else:
        opt.device = torch.device('cpu')

    return opt

import sklearn.metrics
from typing import Dict
import warnings

def cluster_dist_metric(model_out, num_classes=2, gt_label=None):
    '''
    Implements the inter and intra clsuter distance for a sequence of graph
    '''
    
    # Placeholder
    X_labels = []
    
    # Loop and set the labels
    for i in range(num_classes):
        X_label = model_out[gt_label == i]
        
        # Check if torch tensor
        if type(X_label) == torch.Tensor:
            X_label = X_label.data.cpu().numpy()
        
        # Calculate the norm
        h_norm = np.sum(np.square(X_label), axis=1, keepdims=True)
        h_norm[h_norm == 0.] = 1e-3
        X_label = X_label / np.sqrt(h_norm)
        X_labels.append(X_label)

    # Intra cluster distance
    dis_intra = 0.0
    for i in range(num_classes):
        x2 = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
        dists = x2 + x2.T - 2 * np.matmul(X_labels[i], X_labels[i].T)
        dis_intra += np.mean(dists)
    dis_intra /= num_classes
    
    # Inter cluster distance
    dis_inter = 0.0
    for i in range(num_classes-1):
        for j in range(i+1, num_classes):
            x2_i = np.sum(np.square(X_labels[i]), axis=1, keepdims=True)
            x2_j = np.sum(np.square(X_labels[j]), axis=1, keepdims=True)
            dists = x2_i + x2_j.T - 2 * np.matmul(X_labels[i], X_labels[j].T)
            dis_inter += np.mean(dists)
    num_inter = float(num_classes * (num_classes-1) / 2)
    dis_inter /= num_inter

    return dis_intra, dis_inter

def get_unique_probs(x):
    uniqueids = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, unique_inverse, unique_counts = np.unique(uniqueids, return_index=False, return_inverse=True, return_counts=True)
    return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

def Kget_dists(X):
    """Keras code to compute the pairwise distance matrix for a set of
    vectors specifie by the matrix X.
    """
    x2 = np.sum(np.square(X), axis=1, keepdims=True)
    dists = x2 + x2.T - 2 * np.matmul(X, X.T)
    return dists

def entropy_estimator_kl(x, var):
    # KL-based upper bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    #  and Kolchinsky and Tracey, Nonlinear Information Bottleneck, 2017. Eq. 10
    dims, N = float(x.shape[1]), float(x.shape[0])
    dists = Kget_dists(x)
    dists2 = dists / (2*var)
    normconst = (dims / 2.0) * np.log(2 * np.pi * var)
    lprobs = np.log(np.sum(np.exp(-dists2), axis=1)) - np.log(N) - normconst
    h = -np.mean(lprobs)

    return dims/2 + h

def entropy_estimator_bd(x, var):
    # Bhattacharyya-based lower bound on entropy of mixture of Gaussians with covariance matrix var * I
    #  see Kolchinsky and Tracey, Estimating Mixture Entropy with Pairwise Distances, Entropy, 2017. Section 4.
    dims, N = float(x.shape[1]), float(x.shape[0])
    val = entropy_estimator_kl(x,4*var)
    return val + np.log(0.25)*dims/2

def kde_condentropy(x, var):
    # Return entropy of a multivariate Gaussian, in nats
    dims = x.shape[1]
    return (dims/2.0)*(np.log(2*np.pi*var) + 1)


def mi_kde(h, inputdata, var=0.1):
    # function: compute the mutual information between the input and the final representation
    # h: hidden representation at the final layer
    # inputdata: the input attribute matrix X
    # var: noise variance used in estimate the mutual information in KDE    
    nats2bits = float(1.0 / np.log(2))
    h_norm = np.sum(np.square(h), axis=1, keepdims=True)
    h_norm[h_norm == 0.] = 1e-3
    h = h / np.sqrt(h_norm)
    input_norm = np.sum(np.square(inputdata), axis=1, keepdims=True)
    input_norm[input_norm == 0.] = 1e-3
    inputdata = inputdata / np.sqrt(input_norm)

    # the entropy of the input
    entropy_input = entropy_estimator_bd(inputdata, var)

    # compute the entropy of input given the hidden representation at the final layer
    entropy_input_h = 0.
    indices = np.argmax(h, axis=1)
    indices = np.expand_dims(indices, axis=1)
    p_h, unique_inverse_h = get_unique_probs(indices)
    p_h = np.asarray(p_h).T
    for i in range(len(p_h)):
        labelixs = unique_inverse_h==i
        entropy_input_h += p_h[i] * entropy_estimator_bd(inputdata[labelixs, :], var)

    # the mutual information between the input and the hidden representation at the final layer
    MI_HX = entropy_input - entropy_input_h

    return nats2bits*MI_HX

# to build the metrics
def metrics(name: str, **kwargs):
    import torchmetrics
    if name == 'auroc':
        with warnings.catch_warnings(record=True) as w:
            # this one has severe memory leak problems. 
            # do not use
            auroc = torchmetrics.AUROC(pos_label=1, **kwargs)
            return lambda p,t: {
                'auroc': auroc(p,t).item()
            }
    elif name == 'accuracies': 
        conf = torchmetrics.ConfusionMatrix(
            num_classes=2,
            normalize='true', 
            threshold=.5,
        ).to(kwargs.get("device", torch.device("cpu")))

        def _acc(p, t):
            M = conf(p, t)
            tpr = (M[1,1] / M[1].sum()).item()
            tnr = (M[0,0] / M[0].sum()).item()
            return {
                'bacc': (tpr + tnr) / 2.,
                'tpr': tpr,
                'tnr': tnr,
            }

        return _acc
    else:
        raise NotImplementedError


from collections import defaultdict

def train(
    model, data: Dict,
    optimizer, loss_fn,
    accum_gradients=1,
    device=torch.device("cpu"),
    transform=None,
    scheduler=None,
):
    model.train()

    _metrics = {
        "accuracies": metrics("accuracies", device=device),
    }
    _mvals = defaultdict(float)

    # for gradient 
    _n = len(data)

    # Maintain a meter for the loss value
    itr = 0
    skipped = 0
    loss_meter = AverageMeter()
    for _, v in tqdm(enumerate(data), total=len(data), disable=False):
        if v.edge_index.max() < v.num_nodes:
            # Zero the gradient
            optimizer.zero_grad()
            
            # Pass
            out = model(v.x, v.edge_index)
            
            # Find the loss value
            loss = loss_fn(
                out[v.train_mask],
                v.y[v.train_mask]
            )

            # Backprop
            loss.backward()
            optimizer.step()

            # Update the loss meter
            loss_meter.update(loss.item(), 1)
            
            try:
                # Mutual Information and add it to the metric list
                distance_intra, distance_inter = cluster_dist_metric(torch.nn.functional.log_softmax(out[v.train_mask], -1),
                                                                     num_classes=7,
                                                                     gt_label=v.y[v.train_mask])
                dis_ratio = distance_inter / distance_intra
                dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
                _mvals['GDR'] += dis_ratio
                
                # Load the stuff for IIG
                temp_data = v.x.data.cpu().numpy()
                layer_self = out.data.cpu().numpy()
                MI_XiX = mi_kde(layer_self, temp_data, var=0.1)
                _mvals['IIG'] += MI_XiX
                
                # Increment the iterator
                itr += 1
            except Exception as e:
                pass
        else:
            skipped += 1
    
    print(f"Total skipped graphs : {skipped}")
    
    return {
        'loss': round(loss_meter.avg, ndigits=4),
        **{k: round(v / itr, ndigits=4) for k,v in _mvals.items()},
    }

@torch.no_grad()
def evaluate(
    model, loss_fn, data, split, 
    device=torch.device("cpu"),
    transform=None,
):
    # Put in eval mode
    model.eval()

    # Define the metics to calculate
    _metrics = {
        "accuracies": metrics("accuracies", device=device),
    }
    _mvals = defaultdict(float)
    
    # Define the loss meter
    loss_meter = AverageMeter()
        
    # Evaluate
    itr = 0
    for v in data:
        if split == "val":
            split_chosen = v.val_mask
        else:
            split_chosen = v.test_mask
            
        # Check if graphs are good
        if v.edge_index.max() < v.num_nodes:            
            # Get the model output
            out = model(v.x, v.edge_index)
            
            # Find the loss value
            loss = loss_fn(
                out[split_chosen], 
                v.y[split_chosen]
            )
            
            # Update the loss meter
            loss_meter.update(loss.item(), 1)
            
            # Get the metrics
            try:
                # Mutual Information and add it to the metric list
                distance_intra, distance_inter = cluster_dist_metric(torch.nn.functional.log_softmax(out[split_chosen], -1),
                                                                     num_classes=7,
                                                                     gt_label=v.y[split_chosen])
                dis_ratio = distance_inter / distance_intra
                dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
                _mvals['GDR'] += dis_ratio
                
                # Load the stuff for IIG
                temp_data = v.x.data.cpu().numpy()
                layer_self = out.data.cpu().numpy()
                MI_XiX = mi_kde(layer_self, temp_data, var=0.1)
                _mvals['IIG'] += MI_XiX
                        
                # Increment the iterator
                itr += 1
            except Exception as e:
                pass

    return {
        'loss': round(loss_meter.avg, ndigits=4),
        **{k: round(v / itr, ndigits=4) for k,v in _mvals.items()},
    }, None


# build the model, optimizer, loss
def build_model_opt_loss(opt, class_weights):
    # get the class weights
    if opt.model == 'gcn':
        import models.gcn
        model, optimizer, loss = models.gcn.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'gat':
        import models.gat
        model, optimizer, loss = models.gat.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'eresgat':
        import models.eresgat
        model, optimizer, loss = models.eresgat.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'egraphsage':
        import models.egraphsage
        model, optimizer, loss = models.egraphsage.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'gps':
        import models.gps
        model, optimizer, loss = models.gps.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'rggcn':
        import models.rggcn
        model, optimizer, loss = models.rggcn_cora.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'gs':
        import models.gs
        model, optimizer, loss = models.gs.build_model(
            opt,
            class_weights=class_weights,
        )
    elif opt.model == 'dgcn':
        import models.dgcn
        model, optimizer, loss = models.dgcn.build_model(
            opt,
            class_weights=class_weights,
        )
    else:
        raise NotImplementedError

    return model, optimizer, loss

# build the train/eval functions
def build_functions(opt):
    train_args = {
        'accum_gradients': opt.model_accum_grads,
    }
    return (
        train, evaluate, train_args
    )

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main(opt):
    # Fetch the dataset
    path = "./Cora"
    data = Planetoid(path, "Cora", transform=T.NormalizeFeatures())[0]
    num_nodes = data.x.size(0)
    edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
    
    if isinstance(edge_index, tuple):
        data.edge_index = edge_index[0]
    else:
        data.edge_index = edge_index
        
    data = data.to('cuda')
        
    # Define the number of features
    n_features = data.x.shape[-1]
    
    print(f"Total features : {n_features}")

    # Define the input dims for the model
    opt.input_dim = n_features
    print("n_features", opt.input_dim)
    
    # build the model, optimizer, loss
    model, optimizer, loss_fxn = build_model_opt_loss(
        opt, 1.0
    )
    
    # Define the learning rate schedulers
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           patience=2,
                                                           mode="min")

    # some printing
    print(model)

    # number of trainable parameters
    _mp = filter(lambda p: p.requires_grad, model.parameters())
    print (f"Number of trainable parameters: {sum([np.prod(p.size()) for p in _mp])}")

    # build the functions
    (train, evaluate, train_args) = build_functions(opt)

    if opt.tensorboard:
        print ("activate tensorboard")
        from utils import TensorboardWriter
        _external_writer = TensorboardWriter(
            os.path.join(opt.model_path, f"{opt.model}_logs")
        )
    else:
        _external_writer = None

    print ("Training started...")
    os.makedirs(
        opt.model_path, exist_ok=True
    )
    model_save_path = os.path.join(
        opt.model_path, f'{opt.model}_{opt.data_path.split("/")[-1]}_{opt.norm}_{opt.rwpe}_normal.pt'
    )
    print (f"Saving model in \'{model_save_path}\'")
    
    # Define the placeholders for model training
    best = 0
    best_epoch = None
    best_metric = opt.train_best_metric
    
    # Start the model training
    for epoch in range(1, 1 + opt.epochs):
        print("*" * 50 + f"Epoch : {epoch}" + "*" * 50)
        meta = train(
            model=model, data=[data], 
            optimizer=optimizer, loss_fn=loss_fxn, **train_args,
            device=opt.device,
            scheduler=scheduler,
        )

        held_out_results = {}
        for k in ["val", "test"]:
            held_out_results[k], _ = evaluate(
                model, 
                loss_fxn,
                [data],
                k,
                device=opt.device,
            )
        # Print the metric
        meta_a = held_out_results["val"]
        meta_b = held_out_results["test"]
        print(f"Train | Loss : {meta['loss']} | GDR : {meta['GDR']} | IIG : {meta['IIG']}")
        print(f"Valid | Loss : {meta_a['loss']} | GDR : {meta_a['GDR']} | IIG : {meta_a['IIG']}")
        print(f"Test  | Loss : {meta_b['loss']} | GDR : {meta_b['GDR']} | IIG : {meta_b['IIG']}")
        print(f"Learning Rate : {optimizer.param_groups[0]['lr']}")
        print(f"*" * 100)
        
        # Step the scheduler
        scheduler.step(meta_a['loss'])


# build the parser depending on the model
def embelish_model_args(m: str, parser):
    if m == 'gcn':
        import models.gcn
        return models.gcn.args(parser)
    elif m == 'gat':
        import models.gat
        return models.gat.args(parser)
    elif m == 'eresgat':
        import models.eresgat
        return models.eresgat.args(parser)
    elif m == 'egraphsage':
        import models.egraphsage
        return models.egraphsage.args(parser)
    elif m == "gps":
        import models.gps
        return models.gps.args(parser) 
    elif m == "rggcn":
        import models.rggcn_cora
        return models.rggcn_cora.args(parser)
    elif m == "gs":
        import models.gs
        return models.gs.args(parser)
    elif m == "dgcn":
        import models.dgcn
        return models.dgcn.args(parser)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    print("Model training started...........")
    parser = common_args()
    parser = embelish_model_args(
        sys.argv[1], parser
    )

    opt = parse_args(parser)

    print("arguments:")
    print(opt)

    # Seed everything
    seed_everything(10)
    main(opt)

