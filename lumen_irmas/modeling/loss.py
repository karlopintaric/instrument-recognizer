from functools import partial
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss


class FocalLoss(nn.Module):
    """
    Focal Loss implementation.

    This class defines the Focal Loss, which is a variant of the Binary Cross Entropy (BCE) loss that is
    designed to address the problem of class imbalance in binary classification tasks.
    The Focal Loss introduces two hyperparameters, alpha and gamma, to control the balance between easy
    and hard examples during training.

    :param alpha: The balancing parameter between positive and negative examples. A float value between 0 and 1.
        If set to -1, no balancing is applied. Default is 0.25.
    :type alpha: float
    :param gamma: The focusing parameter to control the emphasis on hard examples. A positive integer. Default is 2.
    :type gamma: int
    """

    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super().__init__()
        self.loss_fn = partial(sigmoid_focal_loss, alpha=alpha, gamma=gamma, reduction="mean")

    def forward(self, inputs, targets):
        """
        Compute the Focal Loss.

        :param inputs: The predicted inputs from the model.
        :type inputs: torch.Tensor
        :param targets: The ground truth targets.
        :type targets: torch.Tensor
        :return: The computed Focal Loss.
        :rtype: torch.Tensor
        :raises ValueError: If the inputs and targets have different shapes.
        """

        return self.loss_fn(inputs=inputs, targets=targets)


class HardDistillationLoss(nn.Module):
    """Hard Distillation Loss implementation.

    This class defines the Hard Distillation Loss, which is used for model distillation,
    a technique used to transfer knowledge from a large, complex teacher model to a smaller,
    simpler student model. The Hard Distillation Loss computes the loss by comparing the outputs
    of the student model and the teacher model using a provided loss function. It also introduces a
    threshold parameter to convert the teacher model outputs to binary labels for the distillation process.

    :param teacher: The teacher model used for distillation.
    :type teacher: torch.nn.Module
    :param loss_fn: The loss function used for computing the distillation loss.
    :type loss_fn: torch.nn.Module
    :param threshold: The threshold value used to convert teacher model outputs to binary labels.
        Can be a list or numpy array of threshold values.
    :type threshold: Union[list, np.array]
    :param device: The device to be used for computation. Default is "cuda".
    :type device: str
    """

    def __init__(self, teacher: nn.Module, loss_fn: nn.Module, threshold: Union[list, np.array], device: str = "cuda"):
        super().__init__()
        self.teacher = teacher
        self.loss_fn = loss_fn
        self.threshold = torch.tensor(threshold).to(device)

    def forward(self, inputs, student_outputs, targets):
        """
        Compute the Hard Distillation Loss.

        :param inputs: The input data fed to the student model.
        :type inputs: torch.Tensor
        :param student_outputs: The output predictions from the student model, which consists of
            both classification and distillation outputs.
        :type student_outputs: tuple
        :param targets: The ground truth targets.
        :type targets: torch.Tensor
        :return: The computed Hard Distillation Loss.
        :rtype: torch.Tensor
        :raises ValueError: If the inputs and targets have different shapes.
        """

        outputs_cls, outputs_dist = student_outputs

        teacher_outputs = torch.sigmoid(self.teacher(inputs))
        teacher_labels = (teacher_outputs > self.threshold).float()

        base_loss = self.loss_fn(outputs_cls, targets)
        teacher_loss = self.loss_fn(outputs_dist, teacher_labels)

        return (base_loss + teacher_loss) / 2
