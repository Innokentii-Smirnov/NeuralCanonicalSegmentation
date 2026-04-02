import torch
from torch import Tensor
import torch.nn as nn

class EmbeddingSumEncoder(nn.Module):

  def __init__(self, vocabulary_size: int, embedding_dim: int):
    super(EmbeddingSumEncoder, self).__init__()
    self.embedding = nn.Embedding(vocabulary_size, embedding_dim)

  def forward(self, elements: Tensor) -> Tensor:
    embeddings = self.embedding(elements)
    embedding_sum = torch.sum(embeddings, dim=-2)
    return embedding_sum
