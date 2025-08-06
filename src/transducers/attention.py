import torch
from torch import Tensor
from torch.nn import Module, Linear, Tanh, Softmax

class Attention(Module):

    def __init__(self, first_dim: int, second_dim: int):
        super().__init__()
        self.dense = Linear(first_dim + second_dim, first_dim)
        self.activation = Tanh()
        self.score = Linear(first_dim, 1, bias=False)
        self.softmax = Softmax(dim=-1)

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        # first: N × H₁
        # second: N × L × H₂
        first = first.unsqueeze(1).repeat(1, second.shape[1], 1)
        # similarity: N × L × H₁
        similarity = self.dense(torch.cat([first, second], dim=-1))
        similarity = self.activation(similarity)
        attention = self.score(similarity).squeeze(2)
        probs = self.softmax(attention)
        return probs