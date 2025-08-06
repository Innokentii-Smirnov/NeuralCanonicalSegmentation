from .sequence import SequenceEncoder
from .sequential import SequentialEncoder
from arguments import EncoderArguments
from torch.nn import Module, Dropout
from typing import Optional
from torch import Tensor
from collections import OrderedDict

class TwoLevelSequentialEncoder(Module):

    def __init__(self, lower_encoder_arguments: EncoderArguments,
                 lower_encoding_dropout: float,
                 higher_encoder_arguments: EncoderArguments,
                 features: Optional[OrderedDict[str, tuple[int, int, float]]] = None):

        super().__init__()
        self.lower_encoder = SequenceEncoder(**vars(lower_encoder_arguments))
        self.lower_encoding_dim = self.lower_encoder.output_dim
        self.lower_encoding_dropout = Dropout(lower_encoding_dropout)
        self.higher_encoder = SequentialEncoder(higher_encoder_arguments.hidden_size,
                                                higher_encoder_arguments.num_layers,
                                                higher_encoder_arguments.lstm_dropout,
                                                higher_encoder_arguments.bidirectional,
                                                self.lower_encoding_dim,
                                                features)
        self.output_dim = self.higher_encoder.output_dim

    def forward(self, lower_input: Tensor, higher_input: Tensor):
        batch_size = lower_input.shape[0]
        higher_sequence_length = lower_input.shape[1]
        lower_sequence_length = lower_input.shape[2]

        lower_encoding = self.lower_encoder(lower_input.flatten(0, 1))
        lower_encoding = self.lower_encoding_dropout(lower_encoding)
        higher_encoding = self.higher_encoder(higher_input, lower_encoding.view(batch_size,
                                                                                higher_sequence_length,
                                                                                self.lower_encoding_dim))
        return higher_encoding
