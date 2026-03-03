from typing import TypedDict
from .. import EncoderArguments, CNNArguments

class Hyperparameters(TypedDict):
    encoder_arguments: EncoderArguments
    encoding_dropout: float
    cnn_arguments: CNNArguments
