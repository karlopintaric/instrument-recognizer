import torch
import torch.nn as nn
import timm
from transformers import ASTForAudioClassification, AutoModel, AutoConfig, ASTModel
from .utils import freeze

class StudentAST(nn.Module):
    pass


class StudentClassificationHead(nn.Module):
    pass


class ASTPretrained(nn.Module):
    def __init__(self, n_classes: int, dropout: float):
        super().__init__()
        self.base_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        fc_in = self.base_model.config.hidden_size

        self.classifier = nn.Sequential(
            nn.LayerNorm((fc_in,), eps=1e-12),
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, n_classes))

    def forward(self, x):
        x = self.base_model(x)[1]
        x = self.classifier(x)
        return x

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=[0.1, 0.1]):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # shape = (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


if __name__ == "__main__":
    model = RNN(input_size=128, hidden_size=64, num_layers=3, num_classes=11)
    print(model)