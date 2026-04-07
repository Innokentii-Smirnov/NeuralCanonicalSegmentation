import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional
from collections import OrderedDict
from utils.vocabulary import Vocabulary
from encoders.rnn.sequential import SequentialEncoder
from decoders.rnn.sequential import SequentialDecoder
from arguments import EncoderArguments, NetworkArguments
from encoders.embedding_sum import EmbeddingSumEncoder

class SequenceTransducer(Module):

    def __init__(self,
                 input_vocab_size: int,
                 encoder_arguments: EncoderArguments,
                 encoding_dropout: float,
                 context_dim: int,
                 vocabularies: dict[str, Vocabulary],
                 decoder_arguments: NetworkArguments,
                 device: torch.device,
                 feature_embedding_dim: int | None = None):

        super().__init__()
        self.encoder = SequentialEncoder(
            encoder_arguments['hidden_size'],
            encoder_arguments['num_layers'],
            encoder_arguments['lstm_dropout'],
            encoder_arguments['bidirectional'],
            features = OrderedDict({
                'letters': (
                    input_vocab_size,
                    encoder_arguments['embedding_dim'],
                    encoder_arguments['embedding_dropout']
                )
            })
        )
        if feature_embedding_dim is not None:
            self.feature_encoder: EmbeddingSumEncoder | None = EmbeddingSumEncoder(
              len(vocabularies['features']),
              feature_embedding_dim
            )
            context_dim += feature_embedding_dim
        else:
            self.feature_encoder = None
        self.decoder = SequentialDecoder(input_dim = self.encoder.output_dim,
                                         context_dim = context_dim,
                                         vocabulary = vocabularies['morphon'],
                                         **decoder_arguments,
                                         device = device)

    def encode(self, phon: Tensor, context: Optional[Tensor] = None,
               features: Optional[Tensor] = None):
        """
        :param phon: N × L × V
        :param context: N × C
        :return (encoding, context): (N × L × H₂, N × (C + F))
        """
        encoding = self.encoder({'letters': phon})
        if self.feature_encoder is not None:
          if features is None:
            raise ValueError('This model expects a morphological feature encoding as input.')
          feature_encoding = self.feature_encoder(features)
          if context is not None:
            context = torch.cat((context, feature_encoding), dim=-1)
          else:
            context = feature_encoding
        return encoding, context

    def forward(self, phon: Tensor, morphon: Tensor, context: Optional[Tensor] = None,
                features: Optional[Tensor] = None, **kwargs):
        encoding, context = self.encode(phon, context, features)
        output = self.decoder(encoding, morphon, context)
        return output

    def transduce(self, phon: Tensor, max_output_length: int, context: Optional[Tensor] = None,
                  features: Optional[Tensor] = None):
        encoding, context = self.encode(phon, context, features)
        output = self.decoder.generate(encoding, max_output_length, context)
        return output
