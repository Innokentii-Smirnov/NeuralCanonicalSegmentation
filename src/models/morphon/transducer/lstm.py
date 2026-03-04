import torch
from arguments.transducer.lstm import Hyperparameters
from transducers.rnn.sequence import SequenceTransducer
from utils.vocabulary import Vocabulary

def make_model(vocabularies: dict[str, Vocabulary],
               hyperparameters: Hyperparameters,
               device: torch.device) -> SequenceTransducer:
    model = SequenceTransducer(len(vocabularies["phon"]),
                               context_dim=0,
                               **hyperparameters,
                               decoder_vocabulary=vocabularies['morphon'],
                               device=device)
    return model
