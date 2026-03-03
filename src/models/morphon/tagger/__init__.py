import torch
from utils.vocabulary import Vocabulary
from .cnn import make_model as make_cnn
from .lstm import make_model as make_lstm
from .rcnn import make_model as make_rcnn
from .rcnn_skip_conn import make_model as make_rcnn_skip_conn

def make_model(
    model_type: str,
    vocabularies: dict[str, Vocabulary],
    hyperparameters,
    device: torch.device):
    match model_type:
        case 'CNN':
            return make_cnn(vocabularies, hyperparameters, device)
        case 'LSTM':
            return make_lstm(vocabularies, hyperparameters, device)
        case 'RCNN':
            return make_rcnn(vocabularies, hyperparameters, device)
        case 'RCNN-skip-conn':
            return make_rcnn_skip_conn(vocabularies, hyperparameters, device)
        case _:
            raise ValueError('Unsupported tagger subtype: ' + model_type)
