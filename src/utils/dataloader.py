import torch
from torch import Tensor
import numpy as np
import itertools
from library.iterable import chain_seq
from .padding import pad_tensors, get_sent_tensor

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class FieldBatchDataloader:

    def __init__(self, X, batch_size=32, shuffle=True, sort_by_length=True,
                 length_field=None, device=DEVICE):
        self.X = X
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length
        self.length_field = length_field  ## добавилось
        self.device = device
        self.list_fields = X.list_fields

    def __len__(self):
        return (len(self.X)-1) // self.batch_size + 1

    def __iter__(self):
        if self.sort_by_length:
            # отсортировать индексы по длине объектов [1, ..., 32] -> [7, 4, 15, ...]
            # изменилось взятие длины из поля
            if self.length_field is not None:
                lengths = [len(x[self.length_field]) for x in self.X]
            else:
                lengths = [len(list(x.values())[0]) for x in self.X]
            if self.shuffle:
                order = np.argsort(lengths)
            else:
                order = np.argsort(lengths, kind='stable')
            # сгруппировать в батчи [7, 4, 15, 31, 3, ...] -> [[7, 4, 15, 31], [3, ...], ...]
            batched_order = np.array([order[start:start+self.batch_size]
                                      for start in range(0, len(self.X), self.batch_size)], dtype=object)
            # переупорядочить батчи случайно: [[3, 11, 21, 19], [27, ...], ..., [7, ...], ...]
            if self.shuffle:
                np.random.shuffle(batched_order[:-1])
            # собрать посл-ть индексов: -> [3, 11, 21, 19, 27, ...]
            self.order = np.fromiter(itertools.chain.from_iterable(batched_order), dtype=int)
        else:
            self.order = np.arange(len(self.X))
            if self.shuffle:
                np.random.shuffle(self.order)
        self.idx = 0
        return self

    def __next__(self) -> dict[str, Tensor]:
        if self.idx >= len(self.X):
            raise StopIteration()
        end = min(self.idx + self.batch_size, len(self.X))
        indexes = [self.order[i] for i in range(self.idx, end)]
        batch = dict()
        # перебираем все поля
        for field in self.X[indexes[0]]:
            if field not in self.list_fields:
                batch[field] = pad_tensors([self.X[i][field] for i in indexes]).to(self.device)
            else:
                left = True if field == "letters" else False
                maxlen = max(map(
                    len,
                    chain_seq(self.X[i][field] for i in indexes)
                    )
                )
                batch[field] = pad_tensors(
                    [get_sent_tensor(self.X[i][field], maxlen, left=left)
                    for i in indexes]
                ).to(self.device)
        batch["indexes"] = indexes
        self.idx = end
        return batch
    