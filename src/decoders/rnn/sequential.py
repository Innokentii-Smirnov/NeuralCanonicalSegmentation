import torch
from torch import Tensor, LongTensor
from typing import Optional
from torch.nn import Module, Embedding, LSTM, Linear, LogSoftmax
from utils.vocabulary import Vocabulary
from transducers.attention import Attention

class SequentialDecoder(Module):

    def __init__(self, vocabulary: Vocabulary, embedding_dim: int, input_dim: int,
                 context_dim: int,
                 hidden_size: int, num_layers: int, lstm_dropout: float,
                 device: torch.device):
        super().__init__()
        self.device = device
        self.vocab = vocabulary
        self.num_symbols = len(vocabulary)
        self.embedding = Embedding(self.num_symbols, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_size, input_dim)
        self.rnn = LSTM(embedding_dim + input_dim + context_dim, hidden_size, num_layers,
                        batch_first=True, dropout=lstm_dropout, bidirectional=False)
        self.dense = Linear(hidden_size, self.num_symbols)
        self.log_softmax = LogSoftmax(dim=-1)

    def forward(self, encoder_outputs: Tensor, y: LongTensor, context: Optional[Tensor] = None):
        # encoder_outputs: N × L₂ × H₂
        # context: N × C
        # y: N × L
        batch_size = encoder_outputs.shape[0]
        log_probs = torch.zeros((batch_size, 0, self.num_symbols), dtype=torch.float).to(self.device)
        symbols = torch.zeros((batch_size, 0), dtype=torch.long).to(self.device)
        # symbols will be: N × L
        curr_symbols = torch.full((batch_size, 1), self.vocab.begin, dtype=torch.long).to(self.device)
        h_n = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float).to(self.device)
        c_n = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float).to(self.device)

        for i in range(y.shape[1]):
            embeddings = self.embedding(y[:, i].unsqueeze(1))
            hidden = h_n[-1]
            attention = self.attention(hidden, encoder_outputs).unsqueeze(1)
            weighted = torch.bmm(attention, encoder_outputs)
            # weighted: N × 1 × H₂
            to_concatenate = [embeddings, weighted]
            if context is not None:
                to_concatenate.append(context.unsqueeze(1))
            rnn_input = torch.cat(to_concatenate, dim=-1)
            rnn_output, (h_n, c_n) = self.rnn(rnn_input, (h_n, c_n))
            logits = self.dense(rnn_output)
            curr_log_probs = self.log_softmax(logits)
            _, curr_symbols = torch.max(curr_log_probs, dim=-1)
            log_probs = torch.cat([log_probs, curr_log_probs], dim=1)
            symbols = torch.cat([symbols, curr_symbols], dim=1)

        return {"log_probs": log_probs, "labels": symbols}

    def generate(self, encoder_outputs: Tensor, max_length: int, context: Optional[Tensor] = None):
        # input: N × H
        batch_size = encoder_outputs.shape[0]
        log_probs = torch.zeros((batch_size, 0, self.num_symbols), dtype=torch.float).to(self.device)
        symbols = torch.zeros((batch_size, 0), dtype=torch.long).to(self.device)
        # symbols will be: N × L
        curr_symbols = torch.full((batch_size, 1), self.vocab.begin, dtype=torch.long).to(self.device)
        h_n = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float).to(self.device)
        c_n = torch.zeros((self.rnn.num_layers, batch_size, self.rnn.hidden_size), dtype=torch.float).to(self.device)

        for i in range(max_length):
            embeddings = self.embedding(curr_symbols)
            hidden = h_n[-1]
            attention = self.attention(hidden, encoder_outputs).unsqueeze(1)
            weighted = torch.bmm(attention, encoder_outputs)
            # weighted: N × 1 × H₂
            to_concatenate = [embeddings, weighted]
            if context is not None:
                to_concatenate.append(context.unsqueeze(1))
            rnn_input = torch.cat(to_concatenate, dim=-1)
            rnn_output, (h_n, c_n) = self.rnn(rnn_input, (h_n, c_n))
            logits = self.dense(rnn_output)
            curr_log_probs = self.log_softmax(logits)
            _, curr_symbols = torch.max(curr_log_probs, dim=-1)
            log_probs = torch.cat([log_probs, curr_log_probs], dim=1)
            symbols = torch.cat([symbols, curr_symbols], dim=1)
            sequences_finished = torch.any(symbols == self.vocab.end, dim=1)
            all_sequences_finished = torch.all(sequences_finished)
            if all_sequences_finished:
                break

        return {"log_probs": log_probs, "labels": symbols}