import torch
import torch.nn as nn
from collections import OrderedDict

class McMl(nn.Module):

    def __init__(self, labels_number: OrderedDict[str, int],
                 input_dim: int):
        super().__init__()
        self.categories = list(labels_number.keys())
        self.dense = nn.ModuleDict()
        for category in self.categories:
            self.dense[category] = nn.Linear(input_dim, labels_number[category])
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden):
        log_probs = dict()
        labels = dict()
        for category in self.categories:
            logits = self.dense[category](hidden)
            log_probs[category] = self.log_softmax(logits)
            _, labels[category] = torch.max(log_probs[category], dim=-1)
        return {'log_probs': log_probs, 'labels': labels}
    
