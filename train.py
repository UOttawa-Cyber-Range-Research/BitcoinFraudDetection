# IBM Research Singapore, 2022

# main train script

import warnings
warnings.simplefilter("ignore")
import argparse
from typing import List, Optional
import torch
import sys, os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
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
    parser.add_argument('-train_best_metric', default='bacc')
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
    loss_meter = AverageMeter()
    for cnt, (k, v) in tqdm(enumerate(data.items()), total=len(data), disable=True):
        # Transform the dataset if gps
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

        # _cnt = cnt + 1
        # if (
        #     ((_cnt % accum_gradients) == 0)
        #     or
        #     (_cnt == _n)
        # ):
        #     optimizer.step()

        # Update the loss meter
        loss_meter.update(loss.item(), 1)
        
        try:
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

    return {
        'loss': loss_meter.avg,
        **{k:v / itr for k,v in _mvals.items()},
    }

@torch.no_grad()
def evaluate(
    model, data, labelled, 
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

    # Evaluate
    for k, v in data.items():
        # Collect the model output
        if transform:
            v = transform(v)
        
        # Get the model output
        out = model(v.x, v.edge_index)
        
        # Get the labelled indexes
        _lab = labelled[k]
        
        # Get the metrics
        with warnings.catch_warnings(record=True) as w:
            # Auroc (use the sklearn version)
            _mvals['auroc'] += sklearn.metrics.roc_auc_score(
                v.y[_lab].cpu(),
                torch.sigmoid(out[_lab]).detach().cpu(),
            )
            
            # F1-score (use the sklearn version)
            f1_help = torch.sigmoid(out[_lab]).detach().cpu()
            f1_help = (f1_help > 0.5).long()
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

    return {
        **{k:v / len(data) for k,v in _mvals.items()},
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
        meta = train(
            model=model, data=data[DATA_LABEL_TRAIN], 
            labelled=labelled[DATA_LABEL_TRAIN], 
            optimizer=optimizer, loss_fn=loss, **train_args,
            device=opt.device,
            transform=transform,
            scheduler=scheduler,
        )

        held_out_results = {}
        for k in [DATA_LABEL_VAL, DATA_LABEL_TEST]:
            held_out_results[k], _ = evaluate(
                model, 
                data[k], labelled[k],
                device=opt.device,
                transform=transform,
            )
            
        # Print the metric
        meta_a = held_out_results[DATA_LABEL_VAL]
        meta_b = held_out_results[DATA_LABEL_TEST]
        print("*" * 50 + f"Epoch : {epoch}" + "*" * 50)
        print(f"Train | Loss : {meta['loss']} | Bacc : {meta['bacc']} | Auroc : {meta['auroc']} | F1-score : {meta['f1_score']}")
        print(f"Valid | Bacc : {meta['bacc']} | Auroc : {meta_a['auroc']} | F1-score : {meta_a['f1_score']}")
        print(f"Test  | Bacc : {meta['bacc']} | Auroc : {meta_b['auroc']} | F1-score : {meta_b['f1_score']}")
        print(f"*" * 100)
        
        # Step the scheduler
        scheduler.step(meta['loss'])

        # Extract the held out database
        _metric = held_out_results[DATA_LABEL_VAL][best_metric] 
        
        # Check if the metric has improved over the last best
        if _metric > best:
            best = _metric
            best_test = held_out_results[DATA_LABEL_TEST][best_metric] 
            best_epoch = epoch

            print ("Saving model", best, _metric, best_epoch)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "metrics": {
                        'epoch': epoch,
                        'best_metric': best_metric,
                        **held_out_results,
                    }, 
                    "opt": export_args(opt),
                    "scaler": scaler, 
                    "feature_names": feature_names,
                },
                model_save_path
            )
        
        if _external_writer:
            _external_writer.write(
                "loss/train", meta['loss'], epoch
            )

            for k,v in held_out_results.items():
                for k2 in v:
                    _external_writer.write(
                        f"{k2}/{k}", 
                        v[k2], 
                        epoch
                    )


# build the parser depending on the model
def embelish_model_args(m: str, parser):
    if m == 'gcn':
        import models.gcn
        return models.gcn.args(parser)
    elif m == 'gat':
        import models.gat
        return models.gat.args(parser)
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

