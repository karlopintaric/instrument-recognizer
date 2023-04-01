import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from functools import partial

class FocalLoss(nn.Module):

    def __init__(self, alpha: float=0.25, gamma: int=2):
        super().__init__()
        self.loss_fn = partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")
    
    def forward(self,inputs,targets):
        return self.loss_fn(inputs=inputs, targets=targets)

class WeightedBCELoss(nn.Module):

    def __init__(self, pos_weight=None, n_classes=None):
        super().__init__()
        
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight).repeat(n_classes).half()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)