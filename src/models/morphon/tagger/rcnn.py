import torch
import torch.nn as nn
from torch import Tensor
from collections import OrderedDict
from arguments import EncoderArguments, CNNArguments
from arguments.tagger.rcnn import Hyperparameters
from encoders.rnn.sequential import SequentialEncoder as RecurrentEncoder
from encoders.cnn.sequential import SequentialEncoder as ConvolutionalEncoder
from decoders.mc import Mc
from basic_models.mc import BasicMc
from utils.vocabulary import Vocabulary

class RCNNTagger(nn.Module):

    def __init__(
        self,
        vocabularies: dict[str, Vocabulary],
        encoder_arguments: EncoderArguments,
        encoding_dropout: float,
        cnn_arguments: CNNArguments):
        super(RCNNTagger, self).__init__()
        self.recurrent_encoder = RecurrentEncoder(
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
        self.dropout = nn.Dropout(encoding_dropout)
        self.convolutional_encoder = ConvolutionalEncoder(
          input_dim=self.recurrent_encoder.output_dim,
          **cnn_arguments
        )
        self.decoder = Mc(self.convolutional_encoder.output_dim, len(vocabularies['morphon']))

    def forward(self, phon: Tensor, **kwargs):
        encoding = self.recurrent_encoder({'letters': phon})
        encoding = self.dropout(encoding)
        encoding = self.convolutional_encoder(encoding)
        output = self.decoder(encoding)
        return output

def make_model(vocabularies: dict[str, Vocabulary], hyperparameters: Hyperparameters):
    model = RCNNTagger(
        vocabularies,
        **hyperparameters
    )
    return model
