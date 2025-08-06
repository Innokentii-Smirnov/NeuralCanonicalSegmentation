import torch
import torch.nn as nn
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.dataset import SequenceDataset, SimpleDataset
from utils.dataloader import FieldBatchDataloader

class BasicMc(nn.Module):

    def __init__(self, device: torch.device):
        self.device = device
        # определяем функцию потерь
        self.criterion = nn.NLLLoss(reduction="mean")
        if self.device is not None:
            self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, inputs):
        raise NotImplementedError("You should implement forward pass in your derived class.")

    def train_on_batch(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        loss = self._validate(x, y)
        loss["loss"].backward()
        self.optimizer.step()
        return loss

    def validate_on_batch(self, x, y):
        self.eval()
        with torch.no_grad():
            return self._validate(x, y)

    def _validate(self, x, y):
        if self.device is not None:
            y = y.to(self.device)
        # x -- это словарь
        ## x = {"a": 1, "b": 2}
        ## func(**x) = func(a=1, b=2)
        batch_output = self(**x) #   self.forward(x) = self.__call__(x)
        # классы надо переместить на размерность, идущую после батча
        # log_probs.shape = (B, L, K), y.shape = (B, L)
        loss = self.criterion(batch_output["log_probs"].permute(0, 2, 1), y)
        batch_output["loss"] = loss
        # labels.shape = (B, L)
        return batch_output

    def get_log_probs(self, X: SequenceDataset):
        self.eval()
        dataloader = FieldBatchDataloader(X, device=self.device, batch_size=32)
        answer = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            with torch.no_grad():
                batch_answer = self(**batch)
            # labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            batch_log_probs = batch_answer["log_probs"]
            #_, labels = torch.topk(log_probs, 10)
            for index, sent_log_probs, curr_mask in zip(indexes,
                                                        batch_log_probs,
                                                        batch['mask'].bool().cpu().numpy(),
                                                        strict=True):
                result = sent_log_probs[curr_mask]
                answer[index] = result
        return answer

    def predict(self, X: SimpleDataset) -> list[ndarray]:
        self.eval()
        dataloader = FieldBatchDataloader(X, device=self.device, batch_size=32)
        answer: list[ndarray] = [None] * len(X)
        for batch in tqdm(dataloader):
            indexes = batch["indexes"]
            with torch.no_grad():
                batch_answer = self(batch['phon'])
            labels = batch_answer["labels"].cpu().numpy()
            # probs = batch_answer.cpu().numpy()
            # labels = probs.argmax(axis=-1)
            for index, curr_labels, curr_mask in zip(indexes, labels, batch['mask'].bool().cpu().numpy(), strict=True):
                result = np.take(X.vocabs["morphon"].symbols_, curr_labels[curr_mask])
                answer[index] = result
        return answer
    