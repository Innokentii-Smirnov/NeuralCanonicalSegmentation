from os import path
import json
import torch
from utils.vocabulary import Vocabulary
from .tagger import make_model as make_tagger
from .transducer import make_model as make_transducer
from ..transformer import make_model as make_transformer

def make_model(
    model_type: str,
    model_subtype: str,
    vocabularies: dict[str, Vocabulary],
    device: torch.device,
    use_features: bool,
    hyperparameters = None):
    if hyperparameters is None:
        hyperparameters_directory = model_subtype if use_features else model_subtype + '-pos'
        with open(path.join('default_hyperparameters', model_type, hyperparameters_directory, 'Hyperparameters.json')) as fin:
            hyperparameters = json.load(fin)
    match model_type:
        case 'tagger':
            return make_tagger(model_subtype, vocabularies, hyperparameters, device)
        case 'transducer':
            return make_transducer(model_subtype, vocabularies, hyperparameters, device)
        case 'transformer':
            return make_transformer(vocabularies, hyperparameters, device)
        case _:
            raise ValueError('Unsupported model type: ' + model_type)
