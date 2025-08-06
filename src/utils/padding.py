import torch
from torch import Tensor

def pad_tensor(vec: Tensor, length: int, dim: int, pad_symbol: int, left=False, dtype=torch.long, device=None):
    # vec.shape = [3, 4, 5]
    # length=7, dim=1 -> pad_size = (3, 7-4, 5)
    pad_size = list(vec.shape)
    pad_size[dim] = length - vec.shape[dim]
    padding = torch.ones(*pad_size, dtype=dtype, device=device) * pad_symbol
    if left:
        answer = torch.cat([padding, vec], dim=dim)
    else:
        answer = torch.cat([vec, padding], dim=dim)
    return answer

def pad_tensors(tensors: list[Tensor], pad: int = 0, left: bool = False):
    # дополняет тензоры из tensors до общей максимальной длины символом pad
    tensors = [torch.LongTensor(tensor) for tensor in tensors]
    L = max(tensor.shape[0] for tensor in tensors)
    tensors = [pad_tensor(tensor, L, dim=0, pad_symbol=pad, left=left) for tensor in tensors]
    return torch.stack(tensors, dim=0)

def get_word_tensor(letters: list[int], length: int, left=False) -> Tensor:
    return pad_tensor(torch.LongTensor(letters), length, 0, 0, left)

def get_sent_tensor(words: list[list[int]], length: int, left=False) -> Tensor:
    return torch.stack([get_word_tensor(word, length, left) for word in words])

