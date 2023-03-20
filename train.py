import torch
import numpy as np
import wandb

from utils import parse_config
from dataset import get_loader
from learner import Learner
from models import ASTPretrained, ASTPretrainedBigHead, RNN

SEED = 123
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    
    train_dl = get_loader(config, subset="train")
    valid_dl = get_loader(config, subset="valid")
    
    model = ASTPretrainedBigHead(n_classes=11)

    learn = Learner(train_dl, valid_dl, model, config)

    learn.fit(50)

if __name__=="__main__":

    CONFIG_PATH = "./config.yaml"
    config = parse_config(CONFIG_PATH)

    #run = wandb.init(config=config)
    main(config)
    a=1