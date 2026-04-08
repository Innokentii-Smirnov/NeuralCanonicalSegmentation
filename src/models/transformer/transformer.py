import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from stringutils import string_to_list
from .token_embedding import TokenEmbedding
from .positional_encoding import PositionalEncoding
from .masking import generate_square_subsequent_mask

class Seq2SeqTransformer(nn.Module):
    """
    A sequence to sequence network.
    """
    def __init__(self,
                 src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 device: torch.device):
        src_vocab_size = len(src_vocab)
        tgt_vocab_size = len(tgt_vocab)
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)
        self.begin_token_id = src_vocab.begin
        self.end_token_id = tgt_vocab.end
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_pad = src_vocab.pad
        self.tgt_pad = tgt_vocab.pad

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=self.device).type(torch.bool)

        src_padding_mask = (src == self.src_pad).transpose(0, 1)
        tgt_padding_mask = (tgt == self.tgt_pad).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def greedy_decode(self, src, src_mask, max_len: int, start_symbol: int):
        """
        Generate an output sequence using a greedy algorithm.
        """
        batch_size = src.shape[1]
        memory = self.encode(src, src_mask)
        ys = torch.ones(1, batch_size).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), self.device).type(torch.bool)).to(self.device)
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            _, next_words = torch.max(prob, dim=1)

            ys = torch.cat([ys, next_words.unsqueeze(0)], dim=0)
            if (next_words == self.end_token_id).all():
                break
        return ys
