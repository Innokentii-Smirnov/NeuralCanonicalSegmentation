from typing import TypedDict
from .. import EncoderArguments, NetworkArguments

class Hyperparameters(TypedDict):
    encoder_arguments: EncoderArguments
    encoding_dropout: float
    decoder_arguments: NetworkArguments
