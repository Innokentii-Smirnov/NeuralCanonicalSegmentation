import torch
from torch import Tensor, LongTensor
from torch.nn import Module, ModuleDict, Embedding, Dropout, LSTM, Linear, LogSoftmax
from typing import Optional
from collections import OrderedDict

class SequentialEncoder(Module):

    def __init__(self, hidden_size: int, num_layers: int, lstm_dropout: float,
                 bidirectional: bool,
                 input_dim: int = 0,
                 features: Optional[OrderedDict[str, tuple[int, int, float]]] = None,
                 return_embedding: bool = False):
        super().__init__()
        self.input_dim = input_dim
        total_embedding_dim = 0
        if features is not None:
            self.features = list(features.keys())
            self.embeddings = ModuleDict()
            self.dropouts = dict()
            for feature, (vocab_size, embedding_dim, embedding_dropout) in features.items():
                self.embeddings[feature] = Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.dropouts[feature] = Dropout(embedding_dropout)
                total_embedding_dim += embedding_dim
        else:
            self.features = None
                
        self.rnn = LSTM(total_embedding_dim + self.input_dim, hidden_size, num_layers,
                        batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = self.num_directions * hidden_size
        self.return_embedding = return_embedding

    def forward(self, sequence: dict[str, Tensor], encoding: Optional[Tensor] = None):
        # sequence: dictionary of N × L
        # features: N × L × F
        embeddings = list[Tensor]()
        if self.features is not None:
            for feature in self.features:
                embedding = self.embeddings[feature](sequence[feature])
                embedding = self.dropouts[feature](embedding)
                embeddings.append(embedding)
        if self.input_dim > 0:
            assert encoding is not None
            embeddings.append(encoding)
        inp = torch.cat(embeddings, dim=-1)
        rnn_outputs, (h_n, c_n) = self.rnn(inp)
        if self.return_embedding:
            return rnn_outputs, inp
        else:
            return rnn_outputs
