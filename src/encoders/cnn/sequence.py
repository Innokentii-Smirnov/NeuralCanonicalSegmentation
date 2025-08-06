import torch
import torch.nn as nn
from torch.nn import functional as F

class SequenceEncoder(nn.Module):

    def __init__(self, input_dim: int,
                      n_layers=1, window=5, n_hidden=128, dropout=0.0,
                      use_batch_norm=False):
        self.input_dim = input_dim
        super().__init__()
        self.n_layers = n_layers
        # n_hidden и n_window могут быть своими для каждого слоя,
        # поэтому если передано число, мы его размножаем на все слои.
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden] * self.n_layers
        self.n_hidden = n_hidden
        if isinstance(window, int):
            window = [window] * self.n_layers
        self.window = window
        self.use_batch_norm = use_batch_norm
        # может быть несколько слоёв свёрток
        self.convolutions = nn.ModuleList()
        for i in range(self.n_layers):
            input_dim = self.n_hidden[i-1] if i > 0 else self.input_dim
            convolution = nn.Conv1d(input_dim, self.n_hidden[i], self.window[i],
                                    padding=(self.window[i]-1)//2)
            layer = {
                "convolution": convolution,
                "activation": nn.ReLU(),
                "dropout": nn.Dropout(p=dropout)
            }
            if self.use_batch_norm:
                layer["batch_norm"] = nn.BatchNorm1d(self.n_hidden[i])
            self.convolutions.append(nn.ModuleDict(layer))
        self.output_dim = self.n_hidden[-1]


    def forward(self, embeddings):
        # для свёрточного слоя нужно сделать второй размерностью число каналов
        conv_inputs = embeddings.permute([0, 2, 1])
        for layer in self.convolutions:
            conv_outputs = layer["convolution"](conv_inputs)
            if self.use_batch_norm:
                conv_outputs = layer["batch_norm"](conv_outputs)
            conv_outputs = layer["activation"](conv_outputs)
            conv_outputs = layer["dropout"](conv_outputs)
            conv_inputs = conv_outputs
        conv_outputs = conv_outputs.permute([0, 2, 1])
        max_conv_outputs, _ = torch.max(conv_outputs, dim=1)
        return max_conv_outputs
