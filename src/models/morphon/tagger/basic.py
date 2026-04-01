import torch.nn as nn
from decoders.mc import Mc
from utils.vocabulary import Vocabulary

class BasicTagger(nn.Module):

  def __init__(self, encoding_dim: int, vocabularies: dict[str, Vocabulary]):
    self.decoder = Mc(
        encoding_dim,
        len(vocabularies['morphon'])
    )
