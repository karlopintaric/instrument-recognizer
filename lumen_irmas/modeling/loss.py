import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from functools import partial
from typing import Union, Type


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super().__init__()
        self.loss_fn = partial(
            sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, inputs, targets):
        return self.loss_fn(inputs=inputs, targets=targets)
    
class DistillationLoss(nn.Module):
    
    def __init__(self, teachers: list, loss_fn: Type[nn.Module]):
        super().__init__()
        self.teacher = teachers
        self.loss_fn = loss_fn
    
    def forward(self, inputs, outputs, targets):
        
        outputs_cls, outputs_dist = outputs
        base_loss = self.loss_fn(outputs_cls, targets)

        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)
        
        teacher_loss = self.loss_fn(outputs_dist, teacher_outputs)
        
        return (base_loss + teacher_loss) / 2

