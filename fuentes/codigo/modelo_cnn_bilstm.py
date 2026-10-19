"""CNN-BiLSTM de la tesis y SupCon de una vista con pares supervisados."""
import torch
from torch import nn
from torch.nn import functional as F


class CNNBiLSTMSupCon(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(18, 32, 7, stride=4, padding=3), nn.ReLU(),
                                 nn.Conv1d(32, 64, 5, stride=4, padding=2), nn.ReLU())
        self.bilstm = nn.LSTM(64, 64, num_layers=1, batch_first=True, bidirectional=True)
        self.representation = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        self.classifier = nn.Linear(128, 1)
        self.projection = nn.Linear(128, 64)

    def forward(self, x, contrastive=False):
        features = self.cnn(x).transpose(1, 2)
        sequence, _ = self.bilstm(features)
        h = self.representation(sequence.mean(dim=1))
        logits = self.classifier(h).squeeze(-1)
        if contrastive:
            return logits, F.normalize(self.projection(h), p=2, dim=1)
        return logits


def supervised_contrastive(z, labels, temperature=.1):
    """Promedio por ancla. Excluye auto-similitud y exige pares por clase."""
    if temperature <= 0:
        raise ValueError('Temperatura debe ser positiva')
    n = len(labels)
    diagonal = torch.eye(n, dtype=torch.bool, device=z.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positive.sum(dim=1)
    if (counts == 0).any():
        raise ValueError('Cada clase debe tener al menos dos ejemplos')
    similarity = (z.float() @ z.float().T) / temperature
    denominator = torch.logsumexp(similarity.masked_fill(diagonal, -torch.inf), dim=1)
    log_prob = similarity - denominator[:, None]
    return -((log_prob * positive).sum(dim=1) / counts).mean()
