import torch
from utils.vocabulary import Vocabulary
from .lstm import make_model as make_lstm

def make_model(
    model_type: str,
    vocabularies: dict[str, Vocabulary],
    device: torch.device):
    match model_type:
        case 'LSTM':
            return make_lstm(vocabularies, device)
        case _:
            raise ValueError('Unsupported transducer subtype: ' + model_type)
