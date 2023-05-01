from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

import modeling.loss as loss_module
import modeling.metrics as metrics_module
from modeling.loss import HardDistillationLoss
from modeling.models import freeze, layerwise_lr_decay
from modeling.utils import init_obj


class BaseLearner(ABC):
    """
    Abstract base class for a learner.

    :param train_dl: DataLoader for training data
    :type train_dl: Type[DataLoader]
    :param valid_dl: DataLoader for validation data
    :type valid_dl: Type[DataLoader]
    :param model: Model to be trained
    :type model: Type[nn.Module]
    :param config: Configuration object
    :type config: Any
    """

    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, model: nn.Module, config):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.config = config

    @abstractmethod
    def fit(
        self,
    ):
        """Abstract method for fitting the model."""

        pass

    @abstractmethod
    def _train_epoch(
        self,
    ):
        """Abstract method for training the model for one epoch."""
        pass

    @abstractmethod
    def _test_epoch(
        self,
    ):
        """Abstract method for testing the model for one epoch."""
        pass


class Learner(BaseLearner):
    def __init__(self, train_dl: DataLoader, valid_dl: DataLoader, model: nn.Module, config):
        """A class that inherits from the BaseLearner class and represents a learner object.

        :param train_dl: DataLoader for training data
        :type train_dl: DataLoader
        :param valid_dl: DataLoader for validation data
        :type valid_dl: DataLoader
        :param model: Model to be trained
        :type model: nn.Module
        :param config: Configuration object
        :type config: Any
        """

        super().__init__(train_dl, valid_dl, model, config)

        self.model = torch.nn.DataParallel(module=self.model, device_ids=list(range(config.num_gpus)))
        self.loss_fn = init_obj(self.config.loss, loss_module)
        params = layerwise_lr_decay(self.config, self.model)
        self.optimizer = init_obj(self.config.optimizer, optim, params)
        self.scheduler = init_obj(
            self.config.scheduler,
            optim.lr_scheduler,
            self.optimizer,
            max_lr=[param["lr"] for param in params],
            epochs=self.config.epochs,
            steps_per_epoch=int(np.ceil(len(train_dl) / self.config.num_accum)),
        )

        self.verbose = self.config.verbose
        self.metrics = MetricTracker(self.config.metrics, self.verbose)
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_step = 0
        self.test_step = 0

    def fit(self, model_name: str = "model"):
        """
        Method to train the model.

        :param model_name: Name of the model to be saved, defaults to "model"
        :type model_name: str, optional
        """

        loop = tqdm(range(self.config.epochs), leave=False)

        for epoch in loop:
            train_loss = self._train_epoch()
            val_loss = self._test_epoch()

            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})

            if self.verbose:
                print(f"| EPOCH: {epoch+1} | train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f} |\n")
                self.metrics.display()

        torch.save(self.model.module.state_dict(), f"{model_name}.pth")

    def _train_epoch(self, distill: bool = False):
        """
        Method to perform one epoch of training.

        :param distill: Flag to indicate if knowledge distillation is used, defaults to False
        :type distill: bool, optional
        :return: Average training loss for the epoch
        :rtype: float
        """

        if distill:
            print("Distilling knowledge...", flush=True)

        loop = tqdm(self.train_dl, leave=False)
        self.model.train()

        num_batches = len(self.train_dl)
        train_loss = 0

        for idx, (xb, yb) in enumerate(loop):
            xb = xb.to(self.device)
            yb = yb.to(self.device)

            # forward
            with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=not distill):
                predictions = self.model(xb)

                if distill:
                    loss = self.KDloss_fn(xb, predictions, yb)
                else:
                    loss = self.loss_fn(predictions, yb)

                loss /= self.config.num_accum

            # backward
            self.scaler.scale(loss).backward()
            wandb.log({f"lr_param_group_{i}": lr for i, lr in enumerate(self.scheduler.get_last_lr())})

            if ((idx + 1) % self.config.num_accum == 0) or (idx + 1 == num_batches):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # update loop
            loop.set_postfix(loss=loss.item())
            self.train_step += 1
            wandb.log({"train_loss_per_batch": loss.item(), "train_step": self.train_step})
            train_loss += loss.item()

            if distill:
                if (idx + 1) % 2500 == 0:
                    val_loss = self._test_epoch()
                    wandb.log({"val_loss": val_loss})
                    self.model.train()

        train_loss /= num_batches

        return train_loss

    def _test_epoch(self):
        """
        Method to perform one epoch of validation/testing.

        :return: Average validation/test loss for the epoch
        :rtype: float
        """

        loop = tqdm(self.valid_dl, leave=False)
        self.model.eval()

        num_batches = len(self.valid_dl)
        preds = []
        targets = []
        test_loss = 0

        with torch.no_grad():
            for xb, yb in loop:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb).item()
                self.test_step += 1
                wandb.log({"valid_loss_per_batch": loss, "test_step": self.test_step})
                test_loss += loss

                pred = torch.sigmoid(pred)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        preds, targets = np.array(preds), np.array(targets)
        self.metrics.update(preds, targets)
        test_loss /= num_batches

        return test_loss

class KDLearner(Learner):
    """
    Knowledge Distillation Learner class for training a student model with knowledge distillation.

    :param train_dl: Train data loader
    :type train_dl: DataLoader
    :param valid_dl: Validation data loader
    :type valid_dl: DataLoader
    :param student_model: Student model to be trained
    :type student_model: nn.Module
    :param teacher: Teacher model for knowledge distillation
    :type teacher: nn.Module
    :param thresholds: Thresholds for HardDistillationLoss
    :type thresholds: List[float]
    :param config: Configuration object for training
    :type config: Config
    """

    def __init__(self, train_dl, valid_dl, student_model, teacher, thresholds, config):
        super().__init__(train_dl, valid_dl, student_model, config)

        self.teacher = nn.DataParallel(freeze(teacher).to(self.device))
        self.KDloss_fn = HardDistillationLoss(self.teacher, self.loss_fn, thresholds, self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

    def _train_epoch(self):
        """
        Method to perform one epoch of training with knowledge distillation.

        :return: Average training loss for the epoch
        :rtype: float
        """

        return super()._train_epoch(distill=True)


class MetricTracker:
    """
    Metric Tracker class for tracking evaluation metrics during model validation.
    his class is used to track and display evaluation metrics during model validation.
    It keeps track of the results of the provided metric functions for each validation batch,
    and logs them to Weights & Biases using wandb.log(). The display() method can be used
    to print the tracked metric results, if verbose is set to True during initialization.

    :param metrics: List of metric functions to track
    :type metrics: List[Callable]
    :param verbose: Flag to indicate whether to print the results or not, defaults to True
    :type verbose: bool, optional
    """

    def __init__(self, metrics, verbose: bool = True):
        self.metrics_fn = [getattr(metrics_module, metric) for metric in metrics]
        self.verbose = verbose
        self.result = None

    def update(self, preds, targets):
        """
        Update the metric tracker with the latest predictions and targets.

        :param preds: Model predictions
        :type preds: torch.Tensor
        :param targets: Ground truth targets
        :type targets: torch.Tensor
        """

        self.result = {metric.__name__: metric(preds, targets) for metric in self.metrics_fn}
        wandb.log(self.result)

    def display(self):
        """Display the tracked metric results."""

        for k, v in self.result.items():
            print(f"{k}: {v:.2f}")

def get_preds(data: DataLoader, model: nn.Module, device: str="cpu"):
        
        loop = tqdm(data, leave=False)
        model.eval()

        preds = []
        targets = []

        with torch.no_grad():
            for xb, yb in loop:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                pred = torch.sigmoid(pred)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        preds, targets = np.array(preds), np.array(targets)

        return preds, targets