import torch
from utils.vocabulary import Vocabulary
from .cnn import make_model as make_cnn, CNNTagger
from .lstm import make_model as make_lstm, LSTMTagger
from .rcnn import make_model as make_rcnn, RCNNTagger
from .rcnn_skip_conn import make_model as make_rcnn_skip_conn, RCNNSkipConnTagger

type NeuralTagger = CNNTagger | LSTMTagger | RCNNTagger | RCNNSkipConnTagger

def make_model(
    model_type: str,
    vocabularies: dict[str, Vocabulary],
    hyperparameters,
    device: torch.device) -> NeuralTagger:
    match model_type:
        case 'CNN':
            return make_cnn(vocabularies, hyperparameters)
        case 'LSTM':
            return make_lstm(vocabularies, hyperparameters)
        case 'RCNN':
            return make_rcnn(vocabularies, hyperparameters)
        case 'RCNN-skip-conn':
            return make_rcnn_skip_conn(vocabularies, hyperparameters)
        case _:
            raise ValueError('Unsupported tagger subtype: ' + model_type)
