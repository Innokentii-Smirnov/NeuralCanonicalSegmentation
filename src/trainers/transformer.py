import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from utils.padding import pad_tensor
import numpy as np
from numpy import ndarray
from tqdm.auto import tqdm
from utils.dataset import SimpleDataset
from utils.dataloader import FieldBatchDataloader
from models.transformer.transformer import Seq2SeqTransformer

class TransformerTrainer:
    model: Seq2SeqTransformer
    device: torch.device

    def __init__(self, model: Seq2SeqTransformer, device: torch.device):
        #super(BasicSequenceGenerator, self).__init__()
        self.model = model
        self.device = device
        # определяем функцию потерь
        self.criterion = nn.CrossEntropyLoss()
        if self.device is not None:
            self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    def train_on_batch(self, x, y, mask):
        self.model.train()
        self.optimizer.zero_grad()
        loss, y, mask = self._validate(x, y, mask)
        loss["loss"].backward()
        self.optimizer.step()
        return loss, y, mask

    def validate_on_batch(self, x, y, mask):
        self.model.eval()
        with torch.no_grad():
            return self._validate(x, y, mask)

    def _validate(self, x, y, mask):
        src, tgt = x['phon'].transpose(0, 1), y.transpose(0, 1)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.model.create_mask(src, tgt_input)

        logits = self.model(
          src, tgt_input, src_mask, tgt_mask, src_padding_mask,
          tgt_padding_mask, src_padding_mask
        )

        tgt_out = tgt[1:, :]

        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        _, labels = torch.max(logits, dim=-1)

        batch_output = {
          "loss": loss,
          "labels": labels.transpose(1, 0)
        }
        return batch_output, y[...,1:], mask[...,1:]
