import torch
from torch import Tensor
import torch.nn as nn
from arguments import EncoderArguments, NetworkArguments
from arguments.transducer.lstm import Hyperparameters
from transducers.rnn.sequence import SequenceTransducer
from .basic import BasicMorphonologicalTransducer
from utils.vocabulary import Vocabulary

class MorphonologicalTransducer(BasicMorphonologicalTransducer):

    def __init__(self,
                 vocabularies: dict[str, Vocabulary],
                 encoder_arguments: EncoderArguments,
                 encoding_dropout: float,
                 decoder_arguments: NetworkArguments,
                 device: torch.device,
                 max_output_length: int):

        self.max_output_length = max_output_length
        BasicMorphonologicalTransducer.__init__(self, vocabularies, device)
        self.sequence_transducer = SequenceTransducer(
                                    len(vocabularies["phon"]),
                                    encoder_arguments,
                                    encoding_dropout, 0,
                                    vocabularies['morphon'],
                                    decoder_arguments, device)

    def forward(self, phon: Tensor, morphon: Tensor, generate: bool, **kwargs):
        if generate:
            return self.sequence_transducer.transduce(phon, self.max_output_length)
        else:
            return self.sequence_transducer.forward(phon, morphon)

    def load_state_dict(self, state_dict):
        self.sequence_transducer.load_state_dict(state_dict)

    def state_dict(self):
        return self.sequence_transducer.state_dict()

def make_model(vocabularies: dict[str, Vocabulary],
               hyperparameters: Hyperparameters,
               device: torch.device,
               max_sequence_length: int = 50) -> MorphonologicalTransducer:
    model = MorphonologicalTransducer(
        vocabularies,
        **hyperparameters,
        device=device,
        max_output_length=max_sequence_length
    )
    return model
