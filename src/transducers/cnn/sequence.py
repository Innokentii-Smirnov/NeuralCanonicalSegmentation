import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional
from collections import OrderedDict
from utils.vocabulary import Vocabulary
from encoders.cnn.sequential import SequentialEncoder
from decoders.rnn.sequential import SequentialDecoder
from arguments import CNNArguments, NetworkArguments

class SequenceTransducer(Module):

    def __init__(self,
                 input_vocab_size: int,
                 encoder_arguments: CNNArguments,
                 encoding_dropout: float,
                 context_dim: int,
                 decoder_vocabulary: Vocabulary,
                 decoder_arguments: NetworkArguments,
                 device: torch.device):

        super().__init__()
        self.encoder = SequentialEncoder(
            input_dim = input_vocab_size,
            **encoder_arguments
        )
        self.decoder = SequentialDecoder(input_dim = self.encoder.output_dim,
                                         context_dim = context_dim,
                                         vocabulary = decoder_vocabulary,
                                         **decoder_arguments,
                                         device = device)

    def forward(self, sequence: Tensor, y: Tensor, context: Optional[Tensor] = None):
        # sequence: N × L × V
        # context: N × C
        encoding = self.encoder(sequence)
        # encoding: N × L × H₂
        output = self.decoder(encoding, y, context)
        return output

    def transduce(self, sequence: Tensor, max_output_length: int, context: Optional[Tensor] = None):
        encoding = self.encoder(sequence)
        output = self.decoder.generate(encoding, max_output_length, context)
        return output
