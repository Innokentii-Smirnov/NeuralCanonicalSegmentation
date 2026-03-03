from typing import TypedDict
from .. import EncoderArguments

class Hyperparameters(TypedDict):
    encoder_arguments: EncoderArguments
    encoding_dropout: float
