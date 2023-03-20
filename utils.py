from glob import glob
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import librosa
import yaml
import torch.optim as optim
from types import SimpleNamespace
from torchvision.transforms import Compose

CLASSES = ['tru', 'sax', 'vio', 'gac', 'org', 'cla', 'flu', 'voi', 'gel', 'cel', 'pia']

def get_wav_files(base_path):
    return glob(f"{base_path}/**/*.wav", recursive=True)

class EarlyStopping:
    
    def __init__(self, patience=10, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class Normalize:

    def __init__(self, mean=None, stdev=None):
        
        self.mean = mean
        self.stdev = stdev
    
    def __call__(self,signal):
        
        if self.mean is None and self.stdev is None:
            mean = torch.mean(signal).item()
            stdev = torch.std(signal).item()
        
        return (signal - mean) / (stdev + 1e-8)

def shift_signal(signal, shift_amount):
    return F.pad(signal, (shift_amount, 0), "constant", 0)

def pad_right(signal, amount):
    return F.pad(signal, (0, amount), "constant", 0)

def extract_from_df(df,cols):
    values = []
    for col in cols:
        v = df[col].iloc[0]
        values.append(v)
    return values

def parse_config(config_path):
    with open(config_path) as file:
        return SimpleNamespace(**yaml.safe_load(file))

def init_obj(fn_type, module, *args, **kwargs):
    
    if fn_type is None:
        return None
    
    name = fn_type.get("name")
    
    try:
        fn = getattr(module, name)
    except:
        fn = getattr(nn, name)
    
    fn_args = fn_type.get("args")
    
    if fn_args is not None:
        assert all([k not in fn_args for k in kwargs])
        fn_args.update(kwargs)
        
        return fn(*args, **fn_args)
    else:
        return fn(*args, **kwargs)
    
def diff_lr(config, model):
    parameters = []
    if config.lr_policy['type'] == 'differential':

        for param_dict in config.lr_policy['params']:
            name = param_dict["layers"]
            parameters += [{'params': [p for n, p in model.named_parameters() if name in n],
                    'lr': param_dict['lr']}]
    else:
        parameters = model.parameters()
    
    return parameters

def init_transforms(config, module_name):
    
    if config is None:
        return None

    transforms = []
    for transform in config:
        transforms.append(init_obj(transform, module_name))
    
    return Compose(transforms)

def freeze(model):

    for param in model.parameters():
        param.requires_grad = False
    
    return model

def unfreeze(model):

    for param in model.parameters():
        param.requires_grad = True
    
    return model

if __name__=="__main__":
    import models
    config = parse_config("./config.yaml")
    model = models.RNN(128, 64, 3, 11)
    optimizer = init_obj(config.optimizer, optim, model.parameters())


    
