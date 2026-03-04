import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from arguments import EncoderArguments
from arguments.tagger.lstm import Hyperparameters
from encoders.rnn.sequential import SequentialEncoder
from decoders.mc import Mc
from basic_models.mc import BasicMc
from utils.vocabulary import Vocabulary

class LSTMTagger(nn.Module):

    def __init__(
        self,
        vocabularies: dict[str, Vocabulary],
        encoder_arguments: EncoderArguments,
        encoding_dropout: float):
        super(LSTMTagger, self).__init__()
        self.encoder = SequentialEncoder(
            encoder_arguments['hidden_size'],
            encoder_arguments['num_layers'],
            encoder_arguments['lstm_dropout'],
            encoder_arguments['bidirectional'],
            features = OrderedDict({
                'letters': (
                    len(vocabularies["phon"]),
                    encoder_arguments['embedding_dim'],
                    encoder_arguments['embedding_dropout']
                )
            })
        )
        self.decoder = Mc(self.encoder.output_dim, len(vocabularies['morphon']))


    def forward(self, phon: Tensor, **kwargs):
        encoding = self.encoder({'letters': phon})
        output = self.decoder(encoding)
        return output

def make_model(vocabularies: dict[str, Vocabulary], hyperparameters: Hyperparameters):
    model = LSTMTagger(
        vocabularies,
        **hyperparameters
    )
    return model
