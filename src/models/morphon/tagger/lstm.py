import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from .basic import BasicTagger
from arguments import EncoderArguments
from arguments.tagger.lstm import Hyperparameters
from encoders.rnn.sequential import SequentialEncoder
from utils.vocabulary import Vocabulary

class LSTMTagger(BasicTagger):

    def __init__(
        self,
        vocabularies: dict[str, Vocabulary],
        encoder_arguments: EncoderArguments,
        encoding_dropout: float,
        feature_embedding_dim: int | None = None):
        super(BasicTagger, self).__init__()
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
        super(LSTMTagger, self).__init__(
            self.encoder.output_dim,
            vocabularies,
            feature_embedding_dim
        )


    def encode_phon(self, phon: Tensor) -> Tensor:
        encoding = self.encoder({'letters': phon})
        return encoding

def make_model(vocabularies: dict[str, Vocabulary], hyperparameters: Hyperparameters):
    model = LSTMTagger(
        vocabularies,
        **hyperparameters
    )
    return model
