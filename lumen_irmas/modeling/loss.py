import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from functools import partial
from typing import Union, Type
import numpy as np


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super().__init__()
        self.loss_fn = partial(
            sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, inputs, targets):
        return self.loss_fn(inputs=inputs, targets=targets)


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class HardDistillationLoss(nn.Module):

    def __init__(self, teacher: nn.Module, loss_fn: nn.Module, threshold: Union[list, np.array], device: str = "cuda"):
        super().__init__()
        self.teacher = teacher
        self.loss_fn = loss_fn
        self.threshold = torch.tensor(threshold).to(device)

    def forward(self, inputs, student_outputs, targets):

        outputs_cls, outputs_dist = student_outputs
        
        teacher_outputs = torch.sigmoid(self.teacher(inputs))
        teacher_labels = (teacher_outputs > self.threshold).float()

        base_loss = self.loss_fn(outputs_cls, targets)
        teacher_loss = self.loss_fn(outputs_dist, teacher_labels)

        return (base_loss + teacher_loss) / 2
