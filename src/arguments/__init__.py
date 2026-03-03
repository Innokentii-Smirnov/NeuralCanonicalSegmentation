from typing import TypedDict
from utils.vocabulary import Vocabulary

class NetworkArguments(TypedDict):
    embedding_dim: int
    hidden_size: int
    num_layers: int
    lstm_dropout: float

class EncoderArguments(NetworkArguments):
    vocab_size: int
    embedding_dropout: float
    bidirectional: bool

class DecoderArguments(NetworkArguments):
    vocabulary: Vocabulary

class CNNArguments(TypedDict):
    input_dim: int
    n_layers: int
    window: int | list[int]
    n_hidden: int | list[int]
    dropout: float
    use_batch_norm: bool
