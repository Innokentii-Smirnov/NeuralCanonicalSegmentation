import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.dataset import SequenceDataset, SimpleDataset
from utils.dataloader import FieldBatchDataloader

class McTrainer:

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        # определяем функцию потерь
        self.criterion = nn.NLLLoss(reduction="mean")
        if self.device is not None:
            self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_on_batch(self, x, y, mask):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._validate(x, y)
        loss["loss"].backward()
        self.optimizer.step()
        return loss, y, mask

    def validate_on_batch(self, x, y, mask):
        self.model.eval()
        with torch.no_grad():
            return self._validate(x, y), y, mask

    def _validate(self, x, y):
        if self.device is not None:
            y = y.to(self.device)
        # x -- это словарь
        ## x = {"a": 1, "b": 2}
        ## func(**x) = func(a=1, b=2)
        batch_output = self.model(**x) #   self.forward(x) = self.__call__(x)
        # классы надо переместить на размерность, идущую после батча
        # log_probs.shape = (B, L, K), y.shape = (B, L)
        loss = self.criterion(batch_output["log_probs"].permute(0, 2, 1), y)
        batch_output["loss"] = loss
        # labels.shape = (B, L)
        return batch_output
