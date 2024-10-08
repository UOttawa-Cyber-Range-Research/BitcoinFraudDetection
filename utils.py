# IBM Research Singapore, 2022

from typing import Dict
def export_args(opt, preserved = ['device']):
    return {
        k:v for k,v in 
        opt.__dict__.items() if k not in preserved
    }

def restore_args(opt, data: Dict, preserved = ['device']):
    # this works to restore
    opt.__dict__ = {
        **{k:v for k,v in data.items() if k not in preserved},
        **{n:getattr(opt,n) for n in preserved}
    }
    return opt


# helper function for ensuring endianness
import os
from contextlib import contextmanager

# useful for tqdm null pattern
@contextmanager
def dummy_tqdm(*args, **kwargs):
    
    class dummy:
        def update (self, *args):
            pass
    yield dummy()

import decimal
import json

# for encoding decimal
# https://stackoverflow.com/questions/1960516/python-json-serialize-a-decimal-object
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            return str(o)
        return super(DecimalEncoder, self).default(o)


# for writing torch to a summarywriter

class TensorboardWriter:

    def __init__(self, dir):


        if os.path.exists(dir) == False:
            # try to make the directory if does not exist
            os.mkdir(dir)
        else:
            # then clean the directory
            for x in os.listdir(dir):
                _file = os.path.join(dir, x)
                print ("removing file: ", _file)
                os.remove(_file)

        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(dir)

    # write a sclar
    def write(self, name, value, iter):
        self._writer.add_scalar(name, value, iter)
        
        
import random
import numpy as np
import torch

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