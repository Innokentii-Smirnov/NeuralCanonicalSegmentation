import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional
from collections import OrderedDict
from encoders.rnn.sequential import SequentialEncoder
from decoders.rnn.sequential import SequentialDecoder
from arguments import EncoderArguments, DecoderArguments

class SequenceTransducer(Module):

    def __init__(self, encoder_arguments: EncoderArguments,
                 encoding_dropout: float,
                 context_dim: int,
                 decoder_arguments: DecoderArguments,
                 device: torch.device):

        super().__init__()
        self.encoder = SequentialEncoder(
            encoder_arguments.hidden_size,
            encoder_arguments.num_layers,
            encoder_arguments.lstm_dropout,
            encoder_arguments.bidirectional,
            features = OrderedDict({
                'letters': (
                    encoder_arguments.vocab_size,
                    encoder_arguments.embedding_dim,
                    encoder_arguments.embedding_dropout
                )
            })
        )
        self.decoder = SequentialDecoder(input_dim = self.encoder.output_dim,
                                         context_dim = context_dim,
                                         **vars(decoder_arguments),
                                         device = device)

    def forward(self, sequence: Tensor, y: Tensor, context: Optional[Tensor] = None):
        # sequence: N × L × V
        # context: N × C
        encoding = self.encoder({'letters': sequence})
        # encoding: N × L × H₂
        output = self.decoder(encoding, y, context)
        return output

    def transduce(self, sequence: Tensor, max_output_length: int, context: Optional[Tensor] = None):
        encoding = self.encoder({'letters': sequence})
        output = self.decoder.generate(encoding, max_output_length, context)
        return output