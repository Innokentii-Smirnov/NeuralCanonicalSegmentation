import torch
from utils.vocabulary import Vocabulary
from .tagger import make_model as make_tagger
from .tagger.basic import MorphonologicalTaggerApplier
from .transducer import make_model as make_transducer
from .transducer.basic import MorphonologicalTransducerApplier
from ..transformer import make_model as make_transformer

def make_model(
    model_type: str,
    model_subtype: str,
    vocabularies: dict[str, Vocabulary],
    hyperparameters,
    device: torch.device):
    match model_type:
        case 'tagger':
            return make_tagger(model_subtype, vocabularies, hyperparameters, device)
        case 'transducer':
            return make_transducer(model_subtype, vocabularies, hyperparameters, device)
        case 'transformer':
            return make_transformer(vocabularies, hyperparameters, device)
        case _:
            raise ValueError('Unsupported model type: ' + model_type)

def make_applier(
    model_type: str,
    model,
    vocabularies: dict[str, Vocabulary],
    device: torch.device):
    match model_type:
        case 'tagger':
            return MorphonologicalTaggerApplier(model, vocabularies, device)
        case 'transducer':
            return MorphonologicalTransducerApplier(model, vocabularies, device)
        case 'transformer':
            raise NotImplementedError
        case _:
            raise ValueError('Unsupported model type: ' + model_type)
