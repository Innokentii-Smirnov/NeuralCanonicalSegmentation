import torch
from utils.vocabulary import Vocabulary
from .tagger import make_model as make_tagger
from .transducer import make_model as make_transducer

def make_model(
    model_type: str,
    model_subtype: str,
    vocabularies: dict[str, Vocabulary],
    device: torch.device):
    match model_type:
        case 'tagger':
            return make_tagger(model_subtype, vocabularies, device)
        case 'transducer':
            return make_transducer(model_subtype, vocabularies, device)
        case _:
            raise ValueError('Unsupported model type: ' + model_type)
