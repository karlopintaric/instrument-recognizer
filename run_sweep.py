import torch
import numpy as np
import wandb

from utils import parse_config, unflatten_dot
from dataset import get_loader
from learner import Learner
from models import ASTPretrainedBigHead, ASTPretrainedSmallHead, RNN, ASTWithWeightedLayerPooling, ASTPretrained
from types import SimpleNamespace
import yaml

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    
    wandb.init()
    config = wandb.config
    train_dl = get_loader(config, subset="train")
    valid_dl = get_loader(config, subset="valid")
    
    model = ASTPretrained(n_classes=11)

    learn = Learner(train_dl, valid_dl, model, config)

    learn.fit(4)

if __name__=="__main___":

    CONFIG_PATH = "./sweep_config.yaml"
    with open(CONFIG_PATH) as file:
        config = yaml.safe_load(file)
    
    sweep_id = wandb.sweep(sweep=config)
    wandb.agent(sweep_id, function=main, count=1)
    a=1