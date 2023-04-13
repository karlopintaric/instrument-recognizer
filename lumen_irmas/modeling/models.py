import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig
from warnings import warn
from typing import Type, List
import sys
import copy
from lumen_irmas.modeling.models import freeze


class StudentAST(nn.Module):

    def __init__(self, n_classes: int, hidden_size: int = 384, num_heads: int = 6):
        super().__init__()

        config = ASTConfig(hidden_size=hidden_size,
                           num_attention_heads=num_heads, intermediate_size=hidden_size*2)
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


def LLRD(config, model):
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
    def __init__(self, base_model: nn.Module, weights: List[str]):
        super().__init__()
        weights = [torch.load(weight) for weight in weights]
        self.models = self._load_models(base_model, weights)

    def forward(self, x):
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        return torch.mean(torch.stack(predictions), dim=0)

    def to(self, device: str):
        self.models = [model.to(device) for model in self.models]

    def _load_models(self, base_model, weights):

        models = []

        for weight in weights:
            weight = torch.load(weight)
            model = copy.deepcopy(base_model)
            model.load_state_dict(weight)
            models.append(freeze(model))

        return models


def freeze(model):

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model


def unfreeze(model):

    model.train()
    for param in model.parameters():
        param.requires_grad = True

    return model
