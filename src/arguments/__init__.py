from typing import TypedDict
from utils.vocabulary import Vocabulary

class NetworkArguments(TypedDict):
    embedding_dim: int
    hidden_size: int
    num_layers: int
    lstm_dropout: float

class EncoderArguments(NetworkArguments):
    embedding_dropout: float
    bidirectional: bool

class CNNArguments(TypedDict):
    n_layers: int
    window: int | list[int]
    n_hidden: int | list[int]
    dropout: float
    use_batch_norm: bool
