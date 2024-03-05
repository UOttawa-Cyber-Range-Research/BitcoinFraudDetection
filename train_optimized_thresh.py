import os
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import torch_geometric.transforms as T

# ----------------------------------------------------------
# ARGS
# ----------------------------------------------------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def common_args():
    import argparse
    parser = argparse.ArgumentParser("Train graphical models for bitcoin data")

    parser.add_argument('model') # first positional argument
    parser.add_argument('-model_path', default='data/models')
    parser.add_argument('-rwpe', default=False)
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
    parser.add_argument('-train_best_metric', default='auc_score')
    parser.add_argument('-epochs', type=int, default=12)
    parser.add_argument('-tensorboard', action='store_true', default=False)
    parser.add_argument('-debug', action='store_true', default=False)
    parser.add_argument('-refresh_cache', action='store_true', default=False)
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
            threshold=kwargs.get("thresh", 0.5),
        ).to(kwargs.get("device", torch.device("cpu")))

        def _acc(p, t):
            M = conf(torch.tensor(p), torch.tensor(t).long())
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

def train(
    model, data: Dict, labelled, 
    optimizer, loss_fn,
    accum_gradients=1,
    device=torch.device("cpu"),
    transform=None,
    scheduler=None,
):
    # Put the model in train mode
    model.train()

    # Maintain a meter for the loss value
    skipped = 0
    loss_meter = AverageMeter()
    
    # Evaluate
    list_preds = []
    list_labels = []
    for _, (k, v) in tqdm(enumerate(data.items()), total=len(data), disable=False):
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
            
            # Define the placeholder
            list_preds.append(torch.sigmoid(out[_lab]).view(-1).detach().cpu().numpy())
            list_labels.append(v.y[_lab].view(-1).detach().cpu().numpy())

    # Concat the preds
    list_preds = np.concatenate(list_preds)
    list_labels = np.concatenate(list_labels)
    
    # Compute the auc score
    auc_score = roc_auc_score(list_labels, list_preds)
    print(f"Total skipped graphs : {skipped}")
    
    return {
        'loss': round(loss_meter.avg, ndigits=5),
        'auc_score': round(auc_score, ndigits=5),
    }, list_labels, list_preds

@torch.no_grad()
def evaluate(
    model, data, labelled, 
    device=torch.device("cpu"),
    transform=None,
):
    # Put in eval mode
    model.eval()

    # Evaluate
    list_preds = []
    list_labels = []
    list_x = []
    for k, v in data.items():
        # Collect the model output
        if transform:
            v = transform(v)
        
        # Get the model output
        out = model(v.x, v.edge_index)
        
        # Get the labelled indexes
        _lab = labelled[k]
        
        # Define the placeholder
        list_preds.append(torch.sigmoid(out[_lab]).view(-1).detach().cpu().numpy())
        list_labels.append(v.y[_lab].view(-1).detach().cpu().numpy())
        list_x.append(v.x.detach().cpu().numpy())

    # Concat the preds
    list_preds = np.concatenate(list_preds)
    list_labels = np.concatenate(list_labels)
    list_x = np.concatenate(list_x)
    
    # Return the data
    return {
        "auc_score": round(roc_auc_score(list_labels, list_preds), ndigits=5) 
    }, list_labels, list_preds, list_x


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
from utils import export_args

def main(opt):    
    # Fetch the dataset
    data, labelled, scaler, feature_names, n_features, class_weights = load_data(opt)
    
    # Check if the random walk is there or not
    if opt.rwpe:
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
    
    # build the model, optimizer, loss
    model, optimizer, loss = build_model_opt_loss(
        opt, class_weights
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
        print("activate tensorboard")
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
        opt.model_path, f'{opt.model}.pt'
    )
    print (f"Saving model in \'{model_save_path}\'")
    
    # Define the placeholders for model training
    best = 0
    best_epoch = None
    best_metric = opt.train_best_metric
    
    # Start the model training
    print(f"Transform before : {transform}")
    for epoch in range(1, 1 + opt.epochs):
        train_metric, train_labels, train_preds = train(
            model=model, data=data[DATA_LABEL_TRAIN], 
            labelled=labelled[DATA_LABEL_TRAIN], 
            optimizer=optimizer, loss_fn=loss, **train_args,
            device=opt.device,
            transform=transform,
            scheduler=scheduler,
        )

        # Collect the validation result
        val_metric, _, _, _ = evaluate(
            model,
            data[DATA_LABEL_VAL], labelled[DATA_LABEL_VAL],
            device=opt.device,
            transform=transform,
        )
        
        # Collect the test result
        test_metric, _, _, _ = evaluate(
            model,
            data[DATA_LABEL_TEST], labelled[DATA_LABEL_TEST],
            device=opt.device,
            transform=transform,
        )
            
        # Print the metric
        print("*" * 50 + f"Epoch : {epoch}" + "*" * 50)
        print(f"Train | Loss : {train_metric['loss']} | AUC : {train_metric['auc_score']}")
        print(f"Valid | AUC : {val_metric['auc_score']}")
        print(f"Test  | AUC : {test_metric['auc_score']}")
        print(f"*" * 100)
        
        # Step the scheduler
        scheduler.step(train_metric['loss'])

        # Extract the held out database
        _metric = val_metric[best_metric] 
        
        # Check if the metric has improved over the last best
        if _metric > best:
            best = _metric
            best_epoch = epoch

            print ("Saving model......", best, _metric, best_epoch)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metrics": {
                        'epoch': epoch,
                        'best_metric': best_metric,
                    }, 
                    "opt": export_args(opt),
                    "scaler": scaler, 
                    "feature_names": feature_names,
                },
                model_save_path
            )
            
    # Load the best model again
    model.load_state_dict(torch.load(model_save_path)["state_dict"])
    print("Model weights loaded successfully.......")
    
    # Predict using the model
    _, train_labels, train_preds, train_x = evaluate(
        model,
        data[DATA_LABEL_TRAIN], labelled[DATA_LABEL_TRAIN],
        device=opt.device,
        transform=transform,
    )
    
    _, val_labels, val_preds, val_x = evaluate(
        model,
        data[DATA_LABEL_VAL], labelled[DATA_LABEL_VAL],
        device=opt.device,
        transform=transform,
    )
    
    _, test_labels, test_preds, test_x = evaluate(
        model,
        data[DATA_LABEL_TEST], labelled[DATA_LABEL_TEST],
        device=opt.device,
        transform=transform,
    )
    
    # Calculate the optimal threshold (validation set)
    precision, recall, thresholds = precision_recall_curve(train_labels, train_preds)
    fscore_old = (2 * precision * recall) / (precision + recall)
    fscore_old = fscore_old[:-1]
    fscore = fscore_old[fscore_old > 0]
    thresholds = thresholds[fscore_old > 0]
    ix = np.argmax(fscore)
    
    # Print the scores
    print(f'Best Threshold : {thresholds[ix]} | Best F-score : {fscore[ix]}')
    
    # Define the metrics
    accuracy_metric = metrics("accuracies", thresh=thresholds[ix])
    
    # Print the scores
    train_preds_thresh = (train_preds > thresholds[ix]) * 1
    val_preds_thresh = (val_preds > thresholds[ix]) * 1
    test_preds_thresh = (test_preds > thresholds[ix]) * 1
    
    # Print the metrics for val
    f1_train = round(f1_score(train_labels, train_preds_thresh), ndigits=5)
    roc_auc_score_train = round(roc_auc_score(train_labels, train_preds), ndigits=5)
    accuracy_train = round(accuracy_metric(train_preds, train_labels)["bacc"], ndigits=5)
    print("Stage 1 Complete")
    distance_intra, distance_inter = cluster_dist_metric(train_preds,
                                                         num_classes=2,
                                                         gt_label=train_labels)
    dis_ratio = distance_inter / distance_intra
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
    grd_train = dis_ratio
    print("Stage 2 Complete")
    
    # IIG
    IIG_train = mi_kde(train_preds.reshape(-1, 1), train_x, var=0.1)
    print("Stage 3 Complete")
    
    # Print the metrics for val
    f1_val = round(f1_score(val_labels, val_preds_thresh), ndigits=5)
    roc_auc_score_val = round(roc_auc_score(val_labels, val_preds), ndigits=5)
    accuracy_val = round(accuracy_metric(val_preds, val_labels)["bacc"], ndigits=5)
    distance_intra, distance_inter = cluster_dist_metric(val_preds,
                                                         num_classes=2,
                                                         gt_label=val_labels)
    dis_ratio = distance_inter / distance_intra
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
    grd_val = dis_ratio
    
    # IIG
    IIG_val = mi_kde(val_preds.reshape(-1, 1), val_x, var=0.1)
    
    # Print the metrics for test
    f1_test = round(f1_score(test_labels, test_preds_thresh), ndigits=5)
    roc_auc_score_test = round(roc_auc_score(test_labels, test_preds), ndigits=5)
    accuracy_test = round(accuracy_metric(test_preds, test_labels)["bacc"], ndigits=5)
    distance_intra, distance_inter = cluster_dist_metric(test_preds,
                                                         num_classes=2,
                                                         gt_label=test_labels)
    dis_ratio = distance_inter / distance_intra
    dis_ratio = 1. if np.isnan(dis_ratio) else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("inf") else dis_ratio
    dis_ratio = 1. if float(dis_ratio) == float("-inf") else dis_ratio
    grd_test = dis_ratio
    
    # IIG
    IIG_test = mi_kde(test_preds.reshape(-1, 1), test_x, var=0.1)
    
    # Print the results
    print(f"TRAIN ==> F1 : {f1_train} | ROC : {roc_auc_score_train} | BACC : {accuracy_train} | GDR : {grd_train} | IIG : {IIG_train}")
    print(f"VALID ==> F1 : {f1_val} | ROC : {roc_auc_score_val} | BACC : {accuracy_val} | GDR : {grd_val} | IIG : {IIG_val}")
    print(f"TEST  ==> F1 : {f1_test} | ROC : {roc_auc_score_test} | BACC : {accuracy_test} | GDR : {grd_test} | IIG : {IIG_test}")


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

    parser = common_args()
    parser = embelish_model_args(
        sys.argv[1], parser
    )

    opt = parse_args(parser)

    print ("arguments:")
    print (opt)

    # Seed everything
    seed_everything(10)
    main(opt)

