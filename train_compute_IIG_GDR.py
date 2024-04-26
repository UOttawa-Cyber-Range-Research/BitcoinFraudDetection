import os
import wandb
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.simplefilter("ignore")

import sys
import torch
import pprint
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from metrics.IIG import *
from metrics.GDR import *
from utils import *
import torch_geometric.transforms as T

# ----------------------------------------------------------
# ARGS
# ----------------------------------------------------------
def common_args():
    import argparse
    parser = argparse.ArgumentParser("Train graphical models for bitcoin data")

    parser.add_argument('model') # first positional argument
    parser.add_argument('-model_path', default='data/df')
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
    parser.add_argument('-train_best_metric', default='f1_score')
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
    model, data: Dict, labelled, 
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
    for _, (k, v) in tqdm(enumerate(data.items()), total=len(data), disable=True):
        if v.edge_index.max() < v.num_nodes:
            # Transform the dataset
            if transform:
                v = transform(v)
                
            # Zero the gradient
            optimizer.zero_grad()
            
            # Pass
            out = model(v.x, v.edge_index)
            
            # Find the loss value
            _lab = labelled[k]
            loss = loss_fn(
                out[_lab], 
                v.y[_lab].unsqueeze(dim=-1)
            )

            # Backprop
            loss.backward()
            optimizer.step()

            # Update the loss meter
            loss_meter.update(loss.item(), 1)
            
            try:
                # Mutual Information and add it to the metric list
                distance_intra, distance_inter = cluster_dist_metric(torch.sigmoid(out[_lab]),
                                                                     num_classes=2,
                                                                     gt_label=v.y[_lab])
                distance_gap = distance_inter - distance_intra
                dis_ratio = 1. if distance_gap < 0 else distance_inter / distance_intra
                dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
                _mvals['GDR'] += dis_ratio
                
                # Load the stuff for IIG
                temp_data = v.x.data.cpu().numpy()
                layer_self = out[_lab].data.cpu().numpy()
                MI_XiX = mi_kde(layer_self, temp_data, var=0.1)
                _mvals['IIG'] += MI_XiX
                
                # Used by all metrics
                common = torch.sigmoid(out[_lab]).detach().cpu()
                
                # Auroc
                _mvals['auroc'] += sklearn.metrics.roc_auc_score(
                    v.y[_lab].cpu(),
                    common,
                )
                
                # F1-score
                f1_help = (common > 0.5).long()
                _mvals['f1_score'] += sklearn.metrics.f1_score(
                    v.y[_lab].cpu().long().squeeze(),
                    f1_help.view(-1).squeeze(),
                )

                # Fetch torchmetrics metrics
                for k, m in _metrics.items():
                    for k2, v in m(
                        torch.sigmoid(out[_lab]).squeeze(),
                        v.y_i[_lab].squeeze(),
                    ).items():
                        _mvals[k2] += v
                        
                # Increment the iterator
                itr += 1
            except:
                pass
        else:
            skipped += 1
    
    return {
        'loss': round(loss_meter.avg, ndigits=4),
        **{k: round(v / itr, ndigits=4) for k,v in _mvals.items()},
    }

@torch.no_grad()
def evaluate(
    model, loss_fn, data, labelled, 
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
    for k, v in data.items():
        # Check if graphs are good
        if v.edge_index.max() < v.num_nodes:
            
            # Collect the model output
            if transform:
                v = transform(v)
            
            # Get the model output
            out = model(v.x, v.edge_index)
            
            # Get the labelled indexes
            _lab = labelled[k]
            
            # Find the loss value
            _lab = labelled[k]
            loss = loss_fn(
                out[_lab], 
                v.y[_lab].unsqueeze(dim=-1)
            )
            
            # Update the loss meter
            loss_meter.update(loss.item(), 1)
            
            # Get the metrics
            try:
                # Mutual Information and add it to the metric list
                distance_intra, distance_inter = cluster_dist_metric(torch.sigmoid(out[_lab]),
                                                                    num_classes=2,
                                                                    gt_label=v.y[_lab])
                distance_gap = distance_inter - distance_intra
                dis_ratio = 1. if distance_gap < 0 else distance_inter / distance_intra
                dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
                dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
                _mvals['GDR'] += dis_ratio
                
                # Load the stuff for IIG
                temp_data = v.x.data.cpu().numpy()
                layer_self = out[_lab].data.cpu().numpy()
                MI_XiX = mi_kde(layer_self, temp_data, var=0.1)
                _mvals['IIG'] += MI_XiX
                
                # Used by all metrics
                common = torch.sigmoid(out[_lab]).detach().cpu()
                
                # Auroc
                _mvals['auroc'] += sklearn.metrics.roc_auc_score(
                    v.y[_lab].cpu(),
                    common,
                )
                
                # F1-score
                f1_help = (common > 0.5).long()
                _mvals['f1_score'] += sklearn.metrics.f1_score(
                    v.y[_lab].cpu().long().squeeze(),
                    f1_help.view(-1).squeeze(),
                )

                # see comments in train
                for k, m in _metrics.items():
                    for k2, v in m(
                        torch.sigmoid(out[_lab]).squeeze(),
                        v.y_i[_lab].squeeze(),
                    ).items():
                        _mvals[k2] += v
                        
                # Increment the iterator
                itr += 1
            except Exception as e:
                pass

    return {
        'loss': round(loss_meter.avg, ndigits=4),
        **{k: round(v / itr, ndigits=4) for k,v in _mvals.items()},
    }, None


from datasets.types import DATA_LABEL_TRAIN, DATA_LABEL_VAL, DATA_LABEL_TEST

# load data into tensors
def load_data(opt, **kwargs):
    from datasets.loader import load_data, FEATURE_COLUMNS
    
    _add_f = opt.additional_features 
    if _add_f is None:
        _add_f = []
    data, labelled, scaler, feature_names = load_data(
        opt.data_path,
        opt.device,
        debug=opt.debug,
        semi_supervised=opt.data_disable_semi_sup == False,
        semi_supervised_resample_negs=opt.resample_semi_sup,
        semi_supervised_resample_factor=opt.resample_factor_semi_sup,
        feature_column_names=FEATURE_COLUMNS + _add_f,
        features_dir=opt.features_dir, # used for duckdb queries
        refresh_cache=opt.refresh_cache,
        **kwargs
    )
    n_features = len(feature_names)
    print(feature_names)
    class_weights = opt.class_weight

    return (
        data, labelled, scaler, feature_names,
        n_features, class_weights
    )

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
        model, optimizer, loss = models.rggcn.build_model(
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
    data, labelled, scaler, feature_names, n_features, class_weights = load_data(opt)
    
    # Check if the random walk is there or not
    if opt.rwpe == "true":
        print("Using Random Walks...")
        # Define the transforms here
        transform = T.Compose([T.AddRandomWalkPE(walk_length=opt.walk_length,
                                                attr_name=None)])
        
        # Add the walk features to to the total feature for model construction
        n_features += opt.walk_length
    else:
        print("Not using Random Walks...")
        transform = None
    
    print(f"Total features : {n_features}")

    # Define the input dims for the model
    opt.input_dim = n_features
    print("n_features", opt.input_dim)
    
    # Placeholder for the data
    list_data_train = []
    list_data_valid = []
    list_data_test = []
    
    # Loop over the layers and change the model
    for num_layers in [2, 3, 4, 5, 6, 7, 8]:
        
        # Change the number of layers
        opt.num_layers = num_layers
        
        print("arguments:")
        print(opt)
        
        # Build the model, optimizer, loss
        model, optimizer, loss_fxn = build_model_opt_loss(
            opt, class_weights
        )
        
        # Define the learning rate schedulers
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               patience=1,
                                                               mode="min")

        # Some printing
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
            opt.model_path, f'{opt.model}_{opt.data_path.split("/")[-1]}_{opt.norm}_{opt.rwpe}_normal'
        )
        print (f"Saving model in \'{model_save_path}\'")
        
        # Define the placeholders for model training
        best = float("inf")
        stopping_patience = 2
        
        # Start the model training
        print(f"Transform before : {transform}")
        for _ in tqdm(range(1, 1 + opt.epochs), total=opt.epochs):
            meta = train(
                model=model, data=data[DATA_LABEL_TRAIN], 
                labelled=labelled[DATA_LABEL_TRAIN], 
                optimizer=optimizer, loss_fn=loss_fxn, **train_args,
                device=opt.device,
                transform=transform,
                scheduler=scheduler,
            )

            held_out_results = {}
            for k in [DATA_LABEL_VAL, DATA_LABEL_TEST]:
                held_out_results[k], _ = evaluate(
                    model, 
                    loss_fxn,
                    data[k], labelled[k],
                    device=opt.device,
                    transform=transform,
                )
                
            # Print the metric
            meta_a = held_out_results[DATA_LABEL_VAL]
            meta_b = held_out_results[DATA_LABEL_TEST]
            
            # Step the scheduler
            scheduler.step(meta_a['loss'])

            # Extract the held out database
            _metric = meta_a['loss'] 
            
            # Check if the metric has improved over the last best
            if _metric < best:
                best = _metric
                best_train = meta
                best_test = held_out_results[DATA_LABEL_TEST]
                best_valid = held_out_results[DATA_LABEL_VAL]
                
                # Reset the patience
                stopping_patience = 2
            else:
                # Reduce and break
                stopping_patience -= 1
                if stopping_patience == 0:
                    break
                
        # Save the best data across all the epochs
        list_data_train.append([num_layers, best_train['loss'], best_train['bacc'], best_train['auroc'], best_train['f1_score'], best_train['GDR'], best_train['IIG']])
        list_data_valid.append([num_layers, best_valid['loss'], best_valid['bacc'], best_valid['auroc'], best_valid['f1_score'], best_valid['GDR'], best_valid['IIG']])
        list_data_test.append([num_layers, best_test['loss'], best_test['bacc'], best_test['auroc'], best_test['f1_score'], best_test['GDR'], best_test['IIG']])
        
        # Print for testing
        pprint.pprint(list_data_train)
        pprint.pprint(list_data_valid)
        pprint.pprint(list_data_test)
        
    # Make the dataframe and save the data
    data_columns = ["num_layers", "loss", "bacc", "auroc", "f1_score", "gdr", "iig"]
    metric_train_df = pd.DataFrame(list_data_train, columns=data_columns)
    metric_valid_df = pd.DataFrame(list_data_valid, columns=data_columns)
    metric_test_df = pd.DataFrame(list_data_test, columns=data_columns)
    
    # Save the dataframe
    metric_train_df.to_csv(model_save_path + "_train.csv", index=False)
    metric_valid_df.to_csv(model_save_path + "_valid.csv", index=False)
    metric_test_df.to_csv(model_save_path + "_test.csv", index=False)
    print("Done.....")

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
        import models.rggcn
        return models.rggcn.args(parser)
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

    # Seed everything
    seed_everything(10)
    main(opt)

