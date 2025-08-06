import torch
from torch import LongTensor
from torch.nn import Module, Embedding, Dropout, LSTM

class SequenceEncoder(Module):

    def __init__(self, vocab_size: int, embedding_dim: int, embedding_dropout: float,
                 hidden_size: int, num_layers: int, lstm_dropout: float, bidirectional: bool):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = Dropout(embedding_dropout)
        self.rnn = LSTM(embedding_dim, hidden_size, num_layers,
                        batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = self.num_directions * hidden_size
        self.bidirectional = bidirectional

    def forward(self, sequences: LongTensor):
        batch_size = sequences.shape[0]
        embeddings = self.embedding(sequences)
        embeddings = self.embedding_dropout(embeddings)
        rnn_outputs, (h_n, c_n) = self.rnn(embeddings)
        hidden_states = h_n.view(self.rnn.num_layers, self.num_directions, batch_size, self.rnn.hidden_size)
        last_hidden_state = hidden_states[-1, :, :, :]
        if self.bidirectional:
            forward, reverse = torch.split(last_hidden_state, 1)
            output = torch.cat((forward, reverse), dim=-1).squeeze(0)
        else:
            output = last_hidden_state.squeeze(0)
        return output
