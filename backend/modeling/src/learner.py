from tqdm.autonotebook import tqdm
import torch
import torch.optim as optim
import numpy as np
import wandb
import src.metrics as metrics_module
import src.loss as loss_module
from .utils import init_obj
from .models import freeze, unfreeze, LLRD
from abc import ABC

class BaseLearner(ABC):
    
    def __init__(self, train_dl, valid_dl, model, config):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.config = config
    
    def fit(self,):
        pass
    
    def _train_epoch(self,):
        pass
    
    def _test_epoch(self,):
        pass

class Learner(BaseLearner):

    def __init__(self, train_dl, valid_dl, model, config):
        super().init(train_dl, valid_dl, model, config)

        self.model = torch.nn.DataParallel(self.model)
        self.loss_fn = init_obj(self.config.loss, loss_module)
        params = LLRD(self.config, self.model)
        self.optimizer = init_obj(self.config.optimizer, optim, params)
        self.scheduler = init_obj(self.config.scheduler, optim.lr_scheduler,
                                  self.optimizer,
                                  max_lr=[param["lr"] for param in params],
                                  total_steps=np.ceil(
                                      (self.config.EPOCHS*len(train_dl))/config.num_accum),
                                  epochs=self.config.EPOCHS)

        self.verbose = self.config.verbose
        self.metrics = MetricTracker(self.config.metrics, self.verbose)
        self.scaler = torch.cuda.amp.GradScaler()

        self.train_step = 0
        self.test_step = 0

    def fit(self, model_name: str = "model"):

        best_val_loss = np.inf
        loop = tqdm(range(self.config.EPOCHS), leave=False)

        for epoch in loop:

            train_loss = self._train_epoch()
            val_loss = self._test_epoch()

            wandb.log({"train_loss": train_loss,
                       "val_loss": val_loss,
                       "epoch": epoch+1})

            # self.scheduler.step(val_loss)

            if self.verbose:
                print(
                    f'| EPOCH: {epoch+1} | train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f} |\n')
                self.metrics.display()

            if self.config.save_best_model:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), f"{model_name}.pth")

    def _train_epoch(self):

        loop = tqdm(self.train_dl, leave=False)
        self.model.train()

        num_batches = len(self.train_dl)
        train_loss = 0

        for idx, (Xb, yb) in enumerate(loop):

            Xb = Xb.to(self.device)
            yb = yb.to(self.device)

            # forward
            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.device != "cpu" else torch.bfloat16):
                predictions = self.model(Xb)
                loss = self.loss_fn(predictions, yb)
                loss /= self.config.num_accum

            # backward
            self.scaler.scale(loss).backward()
            wandb.log({f'lr_param_group_{i}': lr for i,
                      lr in enumerate(self.scheduler.get_last_lr())})

            if ((idx + 1) % self.config.num_accum == 0) or (idx + 1 == num_batches):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

            # update loop
            loop.set_postfix(loss=loss.item())
            self.train_step += 1
            wandb.log({"train_loss_per_batch": loss.item(),
                       "train_step": self.train_step})
            train_loss += loss.item()

        train_loss /= num_batches

        return train_loss

    def _test_epoch(self):

        loop = tqdm(self.valid_dl, leave=False)
        self.model.eval()

        num_batches = len(self.valid_dl)
        preds = []
        targets = []
        test_loss = 0

        with torch.no_grad():
            for Xb, yb in loop:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                pred = self.model(Xb)
                loss = self.loss_fn(pred, yb).item()
                self.test_step += 1
                wandb.log({"valid_loss_per_batch": loss,
                           "test_step": self.test_step})
                test_loss += loss

                pred = torch.sigmoid(pred)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        preds, targets = np.array(preds), np.array(targets)
        self.metrics.update(preds, targets)
        test_loss /= num_batches

        return test_loss

    def get_preds(self):

        loop = tqdm(self.valid_dl, leave=False)
        self.model.eval()

        preds = []
        targets = []

        with torch.no_grad():
            for Xb, yb in loop:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                pred = self.model(Xb)
                pred = torch.sigmoid(pred)
                preds.extend(pred.cpu().numpy())
                targets.extend(yb.cpu().numpy())

        preds, targets = np.array(preds), np.array(targets)

        return preds, targets

    def freeze(self):
        self.model = freeze(self.model)

    def unfreeze(self):
        self.model = unfreeze(self.model)        

class MetricTracker:

    def __init__(self, metrics, verbose: bool = True):
        self.metrics_fn = [getattr(metrics_module, metric)
                           for metric in metrics]
        self.verbose = verbose
        self.result = None

    def update(self, preds, targets):

        self.result = {metric.__name__: metric(
            preds, targets) for metric in self.metrics_fn}
        wandb.log(self.result)

    def display(self):
        for k, v in self.result.items():
            print(f'{k}: {v:.2f}')
