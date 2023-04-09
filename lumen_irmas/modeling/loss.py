import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from functools import partial


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super().__init__()
        self.loss_fn = partial(
            sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, inputs, targets):
        return self.loss_fn(inputs=inputs, targets=targets)


