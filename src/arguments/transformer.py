from typing import TypedDict

class Hyperparameters(TypedDict):
    num_encoder_layers: int
    num_decoder_layers: int
    emb_size: int
    nhead: int
    dim_feedforward: int
    dropout: float
