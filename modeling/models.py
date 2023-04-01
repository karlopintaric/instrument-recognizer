import torch
import torch.nn as nn
import timm
from transformers import ASTForAudioClassification, AutoModel, AutoConfig
from modeling.utils import freeze

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average[:, 0]

class ASTWithWeightedLayerPooling(nn.Module):

    def __init__(self, model_name: str="MIT/ast-finetuned-audioset-10-10-0.4593", n_classes: int=11, layer_start_pool: int=9):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        config.update({'output_hidden_states':True})
        self.base_model = freeze(AutoModel.from_pretrained(model_name, config=config))
        
        pooler = WeightedLayerPooling(config.num_hidden_layers,
                                           layer_start=layer_start_pool, 
                                           layer_weights=None)
        
        linear = nn.Linear(config.hidden_size, n_classes)
        self.classifier = nn.Sequential(pooler, linear)

    def forward(self, x):
        x = self.base_model(x)[2]
        x = torch.stack(x)
        x = self.classifier(x)
        return x

def create_pretrained_model(model_name, num_classes, dropout):

    if isinstance(model_name, str):
        base_model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    else:
        base_model = model_name
    nf = base_model.fc.in_features
    
    body = nn.Sequential(*list(base_model.children())[:-1])
    head = nn.Sequential(
                nn.BatchNorm1d(nf),
                nn.Dropout(p=dropout),
                nn.Linear(in_features=nf, out_features=num_classes)
            )
    
    return nn.Sequential(body, head)

class ASTPretrained(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        fc_in = self.model.classifier.dense.in_features
        self.model.classifier.dense = nn.Linear(fc_in, n_classes)
    
    def forward(self, x):
        x = self.model(x).logits
        return x

class ASTPretrainedSmallHead(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        fc_in = model.classifier.dense.in_features
        self.base_model = freeze(nn.Sequential(*list(model.children())[:-1]))

        self.classifier = nn.Sequential(
                        nn.LayerNorm((768,), eps=1e-12),
                        nn.Linear(fc_in, n_classes))
    
    def forward(self, x):
        x = self.base_model(x)[1]
        x = self.classifier(x)
        return x

class ASTPretrainedBigHead(nn.Module):
    def __init__(self, n_classes: int, drop_p: list=[0.25,0.5]):
        super().__init__()
        model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        fc_in = model.classifier.dense.in_features
        self.base_model = freeze(nn.Sequential(*list(model.children())[:-1]))
        self.classifier = nn.Sequential(
                        nn.LayerNorm((768,), eps=1e-12),
                        nn.Dropout(drop_p[0]),
                        nn.Linear(fc_in, 512),
                        nn.LayerNorm((512,), eps=1e-12),
                        nn.Dropout(drop_p[1]),
                        nn.Linear(512,n_classes)
        )
    
    def forward(self, x):
        x = self.base_model(x)[1]
        x = self.classifier(x)
        return x
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=[0.1,0.1]):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0)) # shape = (batch_size, seq_length, hidden_size)
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

if __name__=="__main__":
    model = RNN(input_size=128, hidden_size=64, num_layers=3, num_classes=11)
    print(model)
