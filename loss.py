import torch.nn.functional as F
import torch
import torch.nn as nn

class WeightedFocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, pos_weight=None, n_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight).repeat(n_classes).half().cuda()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none", pos_weight=self.pos_weight)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        return loss.mean()

class WeightedBCELoss:

    def __init__(self, pos_weight=None, n_classes=None):
        super().__init__()
        
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight).repeat(n_classes).half()
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        return F.binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.pos_weight)