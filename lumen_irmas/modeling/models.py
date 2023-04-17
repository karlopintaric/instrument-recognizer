import torch
import torch.nn as nn
from transformers import ASTModel, ASTConfig
from warnings import warn
from typing import List
import copy
import torch.nn.functional as F


class StudentAST(nn.Module):
    """
    A student model for audio classification using the AST architecture.

    :param n_classes: The number of classes to classify.
    :type n_classes: int
    :param hidden_size: The number of units in the hidden layers, defaults to 384.
    :type hidden_size: int, optional
    :param num_heads: The number of attention heads to use, defaults to 6.
    :type num_heads: int, optional
    :param dropout: The dropout probability to use, defaults to 0.25.
    :type dropout: float, optional
    """
    
    def __init__(self, n_classes: int, hidden_size: int = 384, num_heads: int = 6, dropout: float = 0.25):
        super().__init__()

        config = ASTConfig(hidden_size=hidden_size,
                           num_attention_heads=num_heads, intermediate_size=hidden_size*4)
        self.base_model = ASTModel(config=config)
        self.classifier = StudentClassificationHead(hidden_size, n_classes, dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the student model.

        :param x: The input tensor of shape [batch_size, sequence_length, input_dim].
        :type x: torch.Tensor
        :return: The output tensor of shape [batch_size, n_classes].
        :rtype: torch.Tensor
        """
        
        x = self.base_model(x)[0]
        x = self.classifier(x)
        return x


class StudentClassificationHead(nn.Module):
    """
    A classification head for the student model.

    :param emb_size: The size of the embedding.
    :type emb_size: int
    :param n_classes: The number of classes to classify.
    :type n_classes: int
    :param dropout: The dropout probability to use.
    :type dropout: float
    """
    
    def __init__(self, emb_size: int, n_classes: int, dropout: float):
        super().__init__()

        self.cls_head = nn.Linear(emb_size, n_classes)
        self.dist_head = nn.Linear(emb_size, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the classification head.

        :param x: The input tensor of shape [batch_size, emb_size*2].
        :type x: torch.Tensor
        :return: The output tensor of shape [batch_size, n_classes].
        :rtype: torch.Tensor
        """
        
        x_cls, x_dist = x[:, 0], x[:, 1]
        x_cls_head = self.dropout(self.cls_head(x_cls))
        x_dist_head = self.dropout(self.dist_head(x_dist))

        if self.training:
            x = x_cls_head, x_dist_head
        else:
            x = (x_cls_head + x_dist_head) / 2

        return x


class ASTPretrained(nn.Module):
    """
    This class implements a PyTorch module for a pre-trained Audio Set Transformer (AST) model 
    fine-tuned on MIT's dataset for audio event classification.

    :param n_classes: The number of classes for audio event classification.
    :type n_classes: int
    :param dropout: The dropout probability for the fully connected layer, defaults to 0.5.
    :type dropout: float, optional
    :raises ValueError: If n_classes is not positive.
    :raises TypeError: If dropout is not a float or is not between 0 and 1.
    :return: The output tensor of shape [batch_size, n_classes] containing the probabilities of each class.
    :rtype: torch.Tensor
    """
    
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
        """Passes the input tensor through the pre-trained Audio Set Transformer (AST) model followed by a fully connected layer.

        :param x: The input tensor of shape [batch_size, seq_len, num_features].
        :type x: torch.Tensor
        :return: The output tensor of shape [batch_size, n_classes] containing the probabilities of each class.
        :rtype: torch.Tensor
        :raises ValueError: If the shape of x is not [batch_size, seq_len, num_features].
        """
        
        x = self.base_model(x)[1]
        x = self.classifier(x)
        return x


def LLRD(config, model: ASTModel):
    """
    LLRD (Layer-wise Learning Rate Decay) function computes the learning rate for each layer in a deep neural network
    using a specific decay rate and a base learning rate for the optimizer.

    :param config: A configuration object that contains the parameters required for LLRD.
    :type config: Any
    :param model: A PyTorch neural network model.
    :type model: ASTModel

    :raises Warning: If the configuration object does not contain the LLRD parameters.

    :return: A dictionary containing the optimizer parameters (parameters, weight decay, and learning rate) for each layer.
    :rtype: dict
    """
    
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

def freeze(model: nn.Module):
    """
    Freeze function sets the requires_grad attribute to False for all parameters in the given PyTorch neural network model.
    This is used to freeze the weights of the model during training or inference.

    :param model: A PyTorch neural network model.
    :type model: nn.Module

    :return: The same model with requires_grad attribute set to False for all parameters.
    :rtype: nn.Module
    """

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