import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from arguments import EncoderArguments, CNNArguments
from encoders.rnn.sequential import SequentialEncoder as RecurrentEncoder
from encoders.cnn.sequential import SequentialEncoder as ConvolutionalEncoder
from decoders.mc import Mc
from basic_models.mc import BasicMc
from .basic import BasicMorphonologicalTransducer
from utils.vocabulary import Vocabulary

class MorphonologicalTransducer(BasicMorphonologicalTransducer):

    def __init__(
        self,
        vocabularies: dict[str, Vocabulary],
        encoder_arguments: EncoderArguments,
        encoding_dropout: float,
        cnn_arguments: CNNArguments,
        device: torch.device):
        super(BasicMc, self).__init__()
        self.recurrent_encoder = RecurrentEncoder(
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
        self.dropout = nn.Dropout(encoding_dropout)
        cnn_arguments.input_dim = self.recurrent_encoder.output_dim
        self.convolutional_encoder = ConvolutionalEncoder(**vars(cnn_arguments))
        self.decoder = Mc(self.convolutional_encoder.output_dim, len(vocabularies['morphon']))
        super(MorphonologicalTransducer, self).__init__(vocabularies, device)

    def forward(self, phon: Tensor, **kwargs):
        encoding = self.recurrent_encoder({'letters': phon})
        encoding = self.dropout(encoding)
        encoding = self.convolutional_encoder(encoding)
        output = self.decoder(encoding)
        return output

def make_model(vocabularies: dict[str, Vocabulary], device: torch.device):
    model = MorphonologicalTransducer(
        vocabularies,
        EncoderArguments(len(vocabularies["phon"]), 150, 0.1, 400, 1, 0.1, True),
        0.1,
        CNNArguments(None, 3, 5, 192, 0.2, True),
        device
    )
    return model
