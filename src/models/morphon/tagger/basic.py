from torch import Tensor
import torch.nn as nn
from decoders.mc import Mc
from utils.vocabulary import Vocabulary

class BasicTagger(nn.Module):

  def __init__(self, encoding_dim: int, vocabularies: dict[str, Vocabulary]):
    self.decoder = Mc(
        encoding_dim,
        len(vocabularies['morphon'])
    )

  def encode_phon(self, phon: Tensor) -> Tensor:
    """
    Encode a phonetic or graphical representation of the input word.
    """
    raise NotImplementedError

  def forward(self, phon: Tensor, **kwargs):
    phon_encoding = self.encode_phon(phon)
    output = self.decoder(phon_encoding)
    return output
