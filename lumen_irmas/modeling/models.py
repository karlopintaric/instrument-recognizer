import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig
from warnings import warn
from typing import List
import copy
import torch.nn.functional as F


class StudentAST(nn.Module):

    def __init__(self, n_classes: int, hidden_size: int = 384, num_heads: int = 6):
        super().__init__()

        config = ASTConfig(hidden_size=hidden_size,
                           num_attention_heads=num_heads, intermediate_size=hidden_size*4)
        self.base_model = ASTModel(config=config)
        self.classifier = StudentClassificationHead(hidden_size, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.base_model(x)[0]
        x = self.classifier(x)
        return x


class StudentClassificationHead(nn.Module):
    def __init__(self, emb_size: int, n_classes: int):
        super().__init__()

        self.cls_head = nn.Linear(emb_size, n_classes)
        self.dist_head = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor):
        x_cls, x_dist = x[:, 0], x[:, 1]
        x_cls_head = self.cls_head(x_cls)
        x_dist_head = self.dist_head(x_dist)

        if self.training:
            x = x_cls_head, x_dist_head
        else:
            x = (x_cls_head + x_dist_head) / 2

        return x


class ASTPretrained(nn.Module):
    def __init__(self, n_classes: int, dropout: float = 0.5):
        super().__init__()
        self.base_model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593")
        fc_in = self.base_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm((fc_in,), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, n_classes))

    def forward(self, x):
        x = self.base_model(x)[1]
        x = self.classifier(x)
        return x


def LLRD(config, model: ASTModel):
    try:
        config = config.LLRD
    except:
        warn("No LLRD found in config. Learner will use single lr for whole model.")
        return None

    lr = config["base_lr"]
    weight_decay = config["weight_decay"]
    no_decay = ["bias", "layernorm"]
    body = ["embeddings", "encoder.layer"]
    head_params = [(n, p) for n, p in model.named_parameters()
                   if not any(body_param in n for body_param in body)]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in head_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p for n, p in head_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]

    # initialize lrs for every layer
    layers = [getattr(model.module, config["body"]).embeddings] + \
        list(getattr(model.module, config["body"]).encoder.layer)
    layers.reverse()
    for layer in layers:
        lr *= config["lr_decay_rate"]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


class Ensemble(nn.Module):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = models

    def forward(self, x):
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        return torch.mean(torch.stack(predictions), dim=0)

    def to(self, device: str):
        moved_models = [model.to(device) for model in self.models]
        return Ensemble(moved_models)


def freeze(model: nn.Module):

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze(model: nn.Module):

    model.train()
    for param in model.parameters():
        param.requires_grad = True

    return model


def load_models_for_ensemble(base_model: nn.Module, weights: List[str], freeze: bool = True):

    models = []

    for weight in weights:
        weight = torch.load(weight)
        model = copy.deepcopy(base_model)
        model.load_state_dict(weight)
        if freeze:
            model = freeze(model)
        models.append(model)

    return models

def interpolate_params(student: nn.Module, teacher: nn.Module):
    
    new_params = {}

    # Iterate over the parameters in the first model
    for name, param in teacher.base_model.named_parameters():
        # Scale the parameter using interpolate if its shape is different from that of the second model
        target_param = student.base_model.state_dict()[name]
        if param.shape != target_param.shape:
            
            squeeze_count = 0
            permuted = False
            while param.ndim < 4:
                param = param.unsqueeze(0)
                squeeze_count += 1
            
            if param.shape[0] > 1:
                param = param.permute(1,2,3,0)
                target_param = target_param.permute(1,2,3,0)
                permuted = True
                
            if target_param.ndim < 2:
                target_param = target_param.unsqueeze(0)
            
            scaled_param = F.interpolate(param, size=(target_param.shape[-2:]), mode="bilinear")
            
            while squeeze_count > 0:
                scaled_param = scaled_param.squeeze(0)
                squeeze_count -= 1
            
            if permuted:
                scaled_param = scaled_param.permute(-1,0,1,2)
            
        else:
            scaled_param = param
        new_params[name] = scaled_param

    return new_params