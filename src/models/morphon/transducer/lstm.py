import torch
from torch import Tensor
import torch.nn as nn
from arguments import EncoderArguments, DecoderArguments
from transducers.rnn.sequence import SequenceTransducer
from .basic import BasicMorphonologicalTransducer
from utils.vocabulary import Vocabulary

class MorphonologicalTransducer(BasicMorphonologicalTransducer, nn.Module):

    def __init__(self,
                 vocabularies: dict[str, Vocabulary],
                 encoder_arguments: EncoderArguments,
                 encoding_dropout: float,
                 decoder_arguments: DecoderArguments,
                 device: torch.device,
                 max_output_length: int):

        self.max_output_length = max_output_length
        nn.Module.__init__(self)
        self.sequence_transducer = SequenceTransducer(encoder_arguments,
                                    encoding_dropout, 0,
                                    decoder_arguments, device)
        BasicMorphonologicalTransducer.__init__(self, vocabularies, device)

    def forward(self, phon: Tensor, morphon: Tensor, generate: bool, **kwargs):
        if generate:
            return self.sequence_transducer.transduce(phon, self.max_output_length)
        else:
            return self.sequence_transducer.forward(phon, morphon)

    def load_state_dict(self, state_dict):
        self.sequence_transducer.load_state_dict(state_dict)

def make_model(vocabularies: dict[str, Vocabulary],
               device: torch.device,
               max_sequence_length: int = 50,
               encoder_hidden_size: int = 400,
               decoder_hidden_size: int = 800) -> MorphonologicalTransducer:
    model = MorphonologicalTransducer(
        vocabularies,
        EncoderArguments(len(vocabularies["phon"]), 150, 0.1, encoder_hidden_size, 1, 0.1, True),
        0.1,
        DecoderArguments(vocabularies["morphon"], 150, decoder_hidden_size, 1, 0.1),
        device,
        max_sequence_length
    )
    return model
