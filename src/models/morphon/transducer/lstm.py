import torch
from torch import Tensor
from arguments import EncoderArguments, DecoderArguments
from transducers.rnn.sequence import SequenceTransducer
from .basic import BasicMorphonologicalTransducer
from utils.vocabulary import Vocabulary

class MorphonologicalTransducer(BasicMorphonologicalTransducer, SequenceTransducer):

    def __init__(self,
                 vocabularies: dict[str, Vocabulary],
                 encoder_arguments: EncoderArguments,
                 encoding_dropout: float,
                 decoder_arguments: DecoderArguments,
                 device: torch.device,
                 max_output_length: int):

        self.max_output_length = max_output_length
        SequenceTransducer.__init__(self, encoder_arguments,
                                    encoding_dropout, 0,
                                    decoder_arguments, device)
        BasicMorphonologicalTransducer.__init__(self, vocabularies, device)

    def forward(self, phon: Tensor, morphon: Tensor, generate: bool, **kwargs):
        if generate:
            return SequenceTransducer.transduce(self, phon, self.max_output_length)
        else:
            return SequenceTransducer.forward(self, phon, morphon)

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
