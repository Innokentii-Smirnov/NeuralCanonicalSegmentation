import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
from arguments import CNNArguments
from encoders.cnn.sequential import SequentialEncoder as ConvolutionalEncoder
from decoders.mc import Mc
from basic_models.mc import BasicMc
from .basic import BasicMorphonologicalTransducer
from arguments import CNNArguments
from utils.vocabulary import Vocabulary
from utils.dataset import SimpleDataset
from utils.dataloader import DEVICE

class MorphonologicalTransducer(BasicMorphonologicalTransducer):

    def __init__(self,
                vocabularies: dict[str, Vocabulary],
                cnn_arguments: CNNArguments,
                device: torch.device):
        super(BasicMc, self).__init__()
        self.convolutional_encoder = ConvolutionalEncoder(**vars(cnn_arguments))
        self.decoder = Mc(
            self.convolutional_encoder.output_dim,
            len(vocabularies['morphon'])
        )
        super(MorphonologicalTransducer, self).__init__(vocabularies, device)

    def forward(self, phon: Tensor, **kwargs):
        one_hot = F.one_hot(phon, self.convolutional_encoder.input_dim)
        encoding = self.convolutional_encoder(one_hot.float())
        output = self.decoder(encoding)
        return output

def make_model(vocabularies: dict[str, Vocabulary], device: torch.device):
    model = MorphonologicalTransducer(
        vocabularies,
        CNNArguments(len(vocabularies["phon"]), 3, 5, 192, 0.2, True),
        DEVICE
    )
    return model
