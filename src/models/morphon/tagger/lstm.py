import torch
from torch import Tensor
from collections import OrderedDict
from arguments import EncoderArguments
from encoders.rnn.sequential import SequentialEncoder
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
        device: torch.device):
        super(BasicMc, self).__init__()
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
        self.decoder = Mc(self.encoder.output_dim, len(vocabularies['morphon']))
        super(MorphonologicalTransducer, self).__init__(vocabularies, device)

    def forward(self, phon: Tensor, **kwargs):
        encoding = self.encoder({'letters': phon})
        output = self.decoder(encoding)
        return output

def make_model(vocabularies: dict[str, Vocabulary], device: torch.device):
    model = MorphonologicalTransducer(
        vocabularies,
        EncoderArguments(len(vocabularies["phon"]), 150, 0.1, 400, 1, 0.1, True),
        0.1,
        device
    )
    return model