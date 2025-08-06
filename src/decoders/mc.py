import torch
import torch.nn as nn

class Mc(nn.Module):

    def __init__(self, input_dim: int, labels_number: int):
        super().__init__()
        self.dense =  nn.Linear(input_dim, labels_number)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, hidden):
        logits = self.dense(hidden)
        log_probs = self.log_softmax(logits)
        _, labels = torch.max(log_probs, dim=-1)
        return {'log_probs': log_probs, 'labels': labels}