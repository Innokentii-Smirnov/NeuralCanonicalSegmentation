import torch
from utils.vocabulary import Vocabulary
from .tagger import MorphonologicalTaggerApplier
from .transducer import MorphonologicalTransducerApplier

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
