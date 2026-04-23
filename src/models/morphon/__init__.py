from os import path
from typing import Optional, Any
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
    model_directory: str,
    args: Optional[dict[str, Any]] = None):
    hyperparameters_directory = model_subtype + '-pos' if use_features else model_subtype
    hyperparameters_file = path.join(model_directory, 'Hyperparameters.json')
    if path.exists(hyperparameters_file):
      with open(hyperparameters_file, 'r', encoding='utf-8') as fin:
        hyperparameters = json.load(fin)
    else:
      with open(path.join('default_hyperparameters', model_type, hyperparameters_directory, 'Hyperparameters.json')) as fin:
        default_hyperparameters = json.load(fin)
      hyperparameters = default_hyperparameters if args is None else \
        {key: args[key] if key in args and args[key] is not None
                        else default_hyperparameters[key]
        for key in default_hyperparameters}
      with open(hyperparameters_file, 'w', encoding='utf-8') as fout:
        json.dump(hyperparameters, fout)
    match model_type:
        case 'tagger':
            return make_tagger(model_subtype, vocabularies, hyperparameters, device)
        case 'transducer':
            return make_transducer(model_subtype, vocabularies, hyperparameters, device)
        case 'transformer':
            return make_transformer(vocabularies, hyperparameters, device)
        case _:
            raise ValueError('Unsupported model type: ' + model_type)
