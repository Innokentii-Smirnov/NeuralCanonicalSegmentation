import torch
from torch import Tensor
import torch.nn as nn
from decoders.mc import Mc
from utils.vocabulary import Vocabulary
from encoders.embedding_sum import EmbeddingSumEncoder

class BasicTagger(nn.Module):

  def __init__(self, phon_encoding_dim: int, vocabularies: dict[str, Vocabulary], feature_embedding_dim: int | None):
    if feature_embedding_dim is not None:
      self.feature_encoder: EmbeddingSumEncoder | None = EmbeddingSumEncoder(
        len(vocabularies['features']),
        feature_embedding_dim
      )
      encoding_dim = phon_encoding_dim + feature_embedding_dim
    else:
      self.feature_encoder = None
      encoding_dim = phon_encoding_dim
    self.decoder = Mc(
        encoding_dim,
        len(vocabularies['morphon'])
    )

  def encode_phon(self, phon: Tensor) -> Tensor:
    """
    Encode a phonetic or graphical representation of the input word.
    """
    raise NotImplementedError

  def forward(self, phon: Tensor, features: Tensor | None = None, **kwargs):
    phon_encoding = self.encode_phon(phon)
    if self.feature_encoder is not None:
      if features is None:
        raise ValueError('This model expects a morphological feature encoding as input.')
      feature_encoding = self.feature_encoder(features).unsqueeze(dim=1).repeat(1, phon.shape[1], 1)
      encoding = torch.cat((phon_encoding, feature_encoding), dim=-1)
    else:
      encoding = phon_encoding
    output = self.decoder(encoding)
    return output
