import torch
import torch.nn as nn
from torch import Tensor
from entmax import Entmax15Loss
from utils.padding import pad_tensor
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from transducers.rnn.sequence import SequenceTransducer

class SequenceGeneratorTrainer:

    def __init__(self, model: SequenceTransducer, device: torch.device):
        #super(BasicSequenceGenerator, self).__init__()
        self.model = model
        self.device = device
        # определяем функцию потерь
        self.criterion = Entmax15Loss() if self.model.use_entmax else nn.NLLLoss(reduction="mean")
        if self.device is not None:
            self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, *args, **kwargs):
        raise NotImplementedError("You should implement forward pass in your derived class.")

    def train_on_batch(self, x, y, mask):
        self.model.train()
        self.optimizer.zero_grad()
        loss, y, mask = self._validate(x, y, mask, False)
        loss["loss"].backward()
        self.optimizer.step()
        return loss, y, mask

    def validate_on_batch(self, x, y, mask):
        self.model.eval()
        with torch.no_grad():
            return self._validate(x, y, mask, True)

    def _validate(self, x, y, mask, generate: bool):
        #if self.device is not None:
            #y = y.to(self.device)
        # x -- это словарь
        ## x = {"a": 1, "b": 2}
        ## func(**x) = func(a=1, b=2)
        #input = x | {'y': y, 'generate': generate}
        batch_output = self.model(**x, generate=generate) #   self.forward(x) = self.__call__(x)
        # классы надо переместить на размерность, идущую после батча
        # log_probs.shape = (B, L, K), y.shape = (B, L)
        #print(batch_output["log_probs"].shape)
        #print(y.shape)
        if generate:
            if batch_output['log_probs'].shape[-2] > y.shape[-1]:
                # Если модель предсказала лишние символы, будем считать их правильными, если это паддинг
                y = pad_tensor(y, batch_output['log_probs'].shape[-2], -1, 0, device=self.device)
                mask = pad_tensor(mask, batch_output['log_probs'].shape[-2], -1, 1, device=self.device)
            elif batch_output['log_probs'].shape[-2] < y.shape[-1]:
                # Если модель предсказала недостаточно символов, будем считать, что в этих позициях паддинг
                batch_output['log_probs'] = pad_tensor(batch_output['log_probs'], y.shape[-1], -2, 0, dtype=torch.float, device=self.device)
                batch_output['labels'] = pad_tensor(batch_output['labels'], y.shape[-1], -1, 0, device=self.device)
        log_probs = batch_output["log_probs"]
        corr = y[...,1:]
        if self.model.use_entmax:
          pred = log_probs[...,:-1,:]
          loss = self.criterion(pred.reshape(-1, pred.shape[-1]), corr.reshape(-1))
        else:
          pred = log_probs[...,:-1,:].movedim(-1, 1).contiguous()
          loss = self.criterion(pred, corr)
        batch_output["loss"] = loss
        # labels.shape = (B, L)
        batch_output['labels'] = batch_output['labels'][...,:-1]
        return batch_output, corr, mask[...,1:]
