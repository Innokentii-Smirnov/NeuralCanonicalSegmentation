import torch
from torch import Tensor
from torch.nn import Module, Linear, Tanh, Softmax
from entmax import Entmax15

class Attention(Module):

    def __init__(self, use_entmax: bool):
        super().__init__()
        self.to_probs = Entmax15(dim=-1) if use_entmax else Softmax(dim=-1)

    def get_score(self, first: Tensor, second: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, first: Tensor, second: Tensor, mask: Tensor) -> Tensor:
        # first: N × H₁
        # second: N × L × H₂
        score = self.get_score(first, second)
        score = torch.where(mask, score, score.new_full([1], float('-inf')))
        probs = self.to_probs(score)
        return probs

class ConcatAttention(Attention):

    def __init__(self, first_dim: int, second_dim: int, use_entmax: bool):
        super().__init__(use_entmax)
        self.dense = Linear(first_dim + second_dim, first_dim)
        self.activation = Tanh()
        self.score = Linear(first_dim, 1, bias=False)

    def get_score(self, first: Tensor, second: Tensor) -> Tensor:
        # first: N × H₁
        # second: N × L × H₂
        first = first.unsqueeze(1).repeat(1, second.shape[1], 1)
        # similarity: N × L × H₁
        similarity = self.dense(torch.cat([first, second], dim=-1))
        similarity = self.activation(similarity)
        attention = self.score(similarity).squeeze(2)
        return attention

class GeneralAttention(Attention):

    def __init__(self, ht_dim: int, hs_dim: int, use_entmax: bool):
        super().__init__(use_entmax)
        self.dense = Linear(ht_dim, hs_dim, bias=False)

    def get_score(self, ht: Tensor, hs: Tensor) -> Tensor:
        # ht: N × H₁
        # hs: N × L × H₂
        projected_hs = self.dense(hs)
        # projected_hs: N × L × H₁
        score = torch.bmm(projected_hs, ht.unsqueeze(1).transpose(2, 1)).squeeze(2)
        return score
